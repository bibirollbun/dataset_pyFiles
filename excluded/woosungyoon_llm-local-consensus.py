!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


import logging
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer


# Clear existing handlers to prevent duplicate logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('debug.txt', mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Explicitly specify stdout
    ],
    force=True  # Force overwrite existing configuration
)

logger = logging.getLogger()

def debug_print(*args, console=True, sep=' '):
    """Debug print function with flexible output options
    
    Args:
        *args: Arguments to print
        console (bool): If True, output to both console and file; 
                       If False, output to file only
        sep (str): Separator between arguments
    """
    message = sep.join(str(arg) for arg in args)
    
    if console:
        logger.info(message)  # Output to both console and file
    else:
        # File-only output using safer file handling
        with open('debug.txt', 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%H:%M:%S')
            f.write(f"{timestamp} - {message}\n")


import kagglehub

def initialize_gemma(model_name: str = "google/gemma-3n-E2B-it") -> tuple:
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map="cuda:0"
    )
    return model, processor


GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

model, processor = initialize_gemma(GEMMA_PATH)


class GemmaWrapper:
    """Wrapper for Gemma model with multimodal support."""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def generate_response(self, 
                         text_prompt: Optional[str] = None, 
                         image_input: Optional[str] = None, 
                         audio_input: Optional[str] = None, 
                         system_prompt: Optional[str] = None, 
                         max_tokens: int = 512) -> str:
        """Generate response from Gemma model with multimodal inputs."""
        
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        user_content = []
        
        if image_input:
            user_content.append({"type": "image", "image": image_input})
        
        if audio_input:
            user_content.append({"type": "audio", "audio": audio_input})
        
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})
        
        if not user_content:
            raise ValueError("At least one input required: text_prompt, image_input, or audio_input")
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        input_ids = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_len = input_ids["input_ids"].shape[-1]
        input_ids = input_ids.to(self.model.device, dtype=self.model.dtype)
        
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            disable_compile=True,
            streamer=TextStreamer(self.processor, skip_prompt=True)
        )
        
        response = self.processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return response


class GemmaNode:
    """Individual node in the LLM consensus network with persona."""
    
    def __init__(self, node_id: str, gemma_wrapper, node_system_prompt: Optional[str] = None, debug: bool = True):
        self.node_id = node_id
        self.gemma = gemma_wrapper
        self.neighbors: List['GemmaNode'] = []
        self.current_answer: Optional[str] = None
        self.round_opinions: List[Dict[str, str]] = []
        self.node_system_prompt = node_system_prompt
        self.debug = debug
    
    def add_neighbor(self, neighbor_node: 'GemmaNode') -> None:
        """Add a neighbor node to this node's network."""
        if neighbor_node not in self.neighbors and neighbor_node != self:
            self.neighbors.append(neighbor_node)
    
    def clear_neighbors(self) -> None:
        """Remove all neighbor connections."""
        self.neighbors.clear()
    
    def receive_broadcast(self, from_node_id: str, broadcasted_answer: str) -> None:
        """Receive an opinion broadcast from another node."""
        self.round_opinions.append({
            'from': from_node_id,
            'answer': broadcasted_answer
        })
    
    def clear_opinions(self) -> None:
        """Reset round-specific state for next consensus round."""
        self.round_opinions.clear()
    
    def get_current_answer(self) -> Optional[str]:
        """Get the current answer/opinion of this node."""
        return self.current_answer
    
    def solve(self, 
              text_prompt: Optional[str] = None, 
              image_input: Optional[str] = None, 
              audio_input: Optional[str] = None, 
              system_prompt: Optional[str] = None, 
              max_tokens: int = 512) -> str:
        """Solve the problem, considering received opinions from neighbors."""
        
        final_system_prompt = self._build_system_prompt(system_prompt)
        
        if self.round_opinions:
            opinions_context = self._create_opinions_context()
            final_system_prompt = f"{final_system_prompt}\n\n{opinions_context}"
        
        if self.debug:
            debug_print(f"ðŸ’­ {self.node_id} is thinking...")

        self.current_answer = self.gemma.generate_response(
            text_prompt=text_prompt,
            image_input=image_input,
            audio_input=audio_input,
            system_prompt=final_system_prompt,
            max_tokens=max_tokens
        )
                    
        if self.debug:
            debug_print(self.current_answer, console=False)
            debug_print("â”€" * 60)

        return self.current_answer
    
    def _build_system_prompt(self, task_system_prompt: Optional[str]) -> str:
        """Build final system prompt combining persona and task instructions."""
        prompts = []
        
        if self.node_system_prompt:
            prompts.append(self.node_system_prompt)
        
        if task_system_prompt:
            prompts.append(task_system_prompt)
        
        return "\n\n".join(prompts) if prompts else ""
    
    def _create_opinions_context(self) -> str:
        """Create context from collected neighbor opinions."""
        if not self.round_opinions:
            return ""
        
        context_parts = ["Consider these perspectives from your network colleagues:"]
        
        for opinion in self.round_opinions:
            context_parts.append(f"- {opinion['from']}: {opinion['answer']}")
        
        context_parts.append("\nBased on these viewpoints and your own expertise, provide your response.")
        
        return "\n".join(context_parts)


class NetworkInspector:
    """Manages and coordinates consensus rounds between GemmaNodes."""
    
    def __init__(self, debug: bool = True):
        self.nodes: List['GemmaNode'] = []
        self.debug = debug
        self.current_round = 0
    
    def add_node(self, node: 'GemmaNode') -> None:
        """Add a GemmaNode to the network."""
        self.nodes.append(node)
    
    def initialize_all_nodes(self, 
                            text_prompt: Optional[str] = None, 
                            image_input: Optional[str] = None, 
                            audio_input: Optional[str] = None, 
                            system_prompt: Optional[str] = None, 
                            max_tokens: int = 512) -> None:
        """Initialize network by having all nodes independently solve the problem."""
        # Reset all nodes to clean state
        for node in self.nodes:
            node.clear_opinions()
        
        # Each node solves independently
        for node in self.nodes:
            answer = node.solve(
                text_prompt=text_prompt,
                image_input=image_input,
                audio_input=audio_input,
                system_prompt=system_prompt, 
                max_tokens=max_tokens
            )

    def run_consensus_round(self, 
                           text_prompt: Optional[str] = None, 
                           image_input: Optional[str] = None, 
                           audio_input: Optional[str] = None, 
                           system_prompt: Optional[str] = None, 
                           max_tokens: int = 512,
                           max_neighbors: Optional[int] = None) -> None:
        """Execute one consensus round with randomly selected initiator."""
        
        if not self.nodes:
            return
            
        # Select random initiator and target neighbors
        initiator_node = random.choice(self.nodes)
        target_neighbors = initiator_node.neighbors
        if max_neighbors is not None and max_neighbors < len(initiator_node.neighbors):
            target_neighbors = random.sample(initiator_node.neighbors, max_neighbors)
        
        if not target_neighbors or not initiator_node.current_answer:
            return
        
        if self.debug:
            debug_print(f"Selected initiator: {initiator_node.node_id}")
            neighbor_names = [n.node_id for n in target_neighbors]
            debug_print(f"Broadcasting to: {neighbor_names}")
        
        # Broadcast â†’ Update â†’ Collect â†’ Synthesize â†’ Broadcast â†’ Update
        
        # First broadcast: Initiator shares opinion with neighbors
        # Neighbors update their opinions based on initiator's input
        self._send_opinion_to_neighbors(initiator_node, target_neighbors)
        
        if self.debug:
            debug_print("ðŸ”„ Phase 1: Neighbors updating opinions based on initiator...")
        self._neighbors_update_opinions(target_neighbors, text_prompt, image_input, audio_input, system_prompt, max_tokens)
        
        # Initiator collects updated opinions from neighbors
        # Initiator synthesizes collected opinions into new consensus
        self._collect_neighbor_opinions(initiator_node, target_neighbors)
        
        if self.debug:
            debug_print(f"{initiator_node.node_id} synthesizing collected opinions...")
        initiator_node.solve(text_prompt, image_input, audio_input, system_prompt, max_tokens)
        
        # Second broadcast: Initiator shares refined opinion back to neighbors
        # Neighbors update final opinions and reset for next round
        self._send_opinion_to_neighbors(initiator_node, target_neighbors)
        
        if self.debug:
            debug_print("ðŸ”„ Phase 2: Neighbors final opinion update...")
        self._neighbors_update_opinions(target_neighbors, text_prompt, image_input, audio_input, system_prompt, max_tokens)
    
    def _send_opinion_to_neighbors(self, initiator_node: 'GemmaNode', target_neighbors: List['GemmaNode']) -> None:
        """Broadcast initiator's opinion to selected neighbors."""
        if not initiator_node.current_answer:
            return
            
        for neighbor in target_neighbors:
            neighbor.receive_broadcast(initiator_node.node_id, initiator_node.current_answer)
    
    def _collect_neighbor_opinions(self, initiator_node: 'GemmaNode', target_neighbors: List['GemmaNode']) -> None:
        """Gather opinions from neighbors back to the initiator."""
        for neighbor in target_neighbors:
            if neighbor.current_answer:
                initiator_node.receive_broadcast(neighbor.node_id, neighbor.current_answer)
    
    def _neighbors_update_opinions(self, 
                                  target_neighbors: List['GemmaNode'],
                                  text_prompt: Optional[str],
                                  image_input: Optional[str],
                                  audio_input: Optional[str],
                                  system_prompt: Optional[str],
                                  max_tokens: int) -> None:
        """Have neighbors solve with received input and reset state."""
        for neighbor in target_neighbors:
            neighbor.solve(text_prompt, image_input, audio_input, system_prompt, max_tokens)
            neighbor.clear_opinions()
    
    def get_all_opinions(self) -> Dict[str, Optional[str]]:
        """Return current opinions from all nodes in the network."""
        return {node.node_id: node.current_answer for node in self.nodes}


class GeneralType(Enum):
    LOYAL = "loyal"
    TRAITOR = "traitor"

@dataclass
class GeneralCharacter:
    """General character information"""
    name: str
    type: GeneralType
    persona: str


class NetworkTopology(ABC):
    """Abstract base class for network topologies"""
    
    @abstractmethod
    def create_connections(self, nodes: List) -> None:
        """Create connections between nodes"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get topology description"""
        pass

class FullyConnectedTopology(NetworkTopology):
    """Fully connected topology - all nodes connected to each other"""
    
    def create_connections(self, nodes: List) -> None:
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    node1.add_neighbor(node2)
    
    def get_description(self) -> str:
        return "Fully Connected (all generals can communicate with each other)"

class PartialMeshTopology(NetworkTopology):
    """Partial mesh topology - each node connected to k neighbors"""
    
    def __init__(self, k: int = 2):
        self.k = k  # Number of connections per node
    
    def create_connections(self, nodes: List) -> None:
        n = len(nodes)
        for i, node in enumerate(nodes):
            # Connect to next k nodes (circular)
            for j in range(1, self.k + 1):
                neighbor_index = (i + j) % n
                neighbor_node = nodes[neighbor_index]
                node.add_neighbor(neighbor_node)
                # Add reverse connection for bidirectional communication
                neighbor_node.add_neighbor(node)
    
    def get_description(self) -> str:
        return f"Partial Mesh (each general communicates with {self.k} neighbors)"


class GeneralNodeCreator:
    """General node creation and network configuration"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def create_character_set(self, loyal_count: int, traitor_count: int) -> List[GeneralCharacter]:
        """Create character set"""
        characters = []
        
        # Loyal generals (animal-based names)
        loyal_templates = [
            ("General_Deerius", 
            "You are a LOYAL general with a cautious, analytical mindset. You believe thorough preparation "
            "and complete intelligence are essential for military success. You tend to prefer defensive strategies "
            "and detailed reconnaissance before major operations. You value consensus-building and careful deliberation."),
            
            ("General_Rabbitson",
            "You are a LOYAL general with extensive battlefield experience. You believe in swift, decisive action "
            "when opportunities arise. You trust your military instincts and prefer to act before the enemy "
            "can regroup or reinforce. You value boldness balanced with tactical wisdom."),
            
            ("General_Alpacus", 
            "You are a LOYAL general who excels at logistics and supply management. You always consider "
            "the practical aspects of military operations - supply lines, troop morale, weather conditions. "
            "You tend to support operations that are well-supplied and logistically sound."),
            
            ("General_Foxworth",
            "You are a LOYAL general with strong diplomatic instincts. You consider the political implications "
            "of military actions and prefer strategies that minimize casualties while maximizing strategic gains. "
            "You often seek compromise solutions that address multiple concerns."),
            
            ("General_Wolfram",
            "You are a LOYAL general who specializes in cavalry tactics and mobile warfare. You favor strategies "
            "that leverage speed, maneuverability, and surprise. You believe the best defense is often a good offense "
            "and prefer to keep the enemy off-balance."),
            
            ("General_Elkridge",
            "You are a LOYAL general with expertise in intelligence and reconnaissance. You always want more "
            "information before making decisions and are skilled at detecting inconsistencies in reports. "
            "You prefer strategies based on solid intelligence rather than assumptions."),
            
            ("General_Bearmont",
            "You are a LOYAL general focused on troop welfare and morale. You consider the human cost of operations "
            "and prefer strategies that protect your soldiers while achieving objectives. You're sensitive to "
            "factors affecting unit cohesion and fighting spirit.")
        ]
                
        # Traitor generals (hedgehog-based names)
        traitor_templates = [
            ("General_Hedgius",
            "You are a TRAITOR general, but you genuinely believe you're serving a greater good. You think "
            "the current empire leadership is corrupt and incompetent. Your secret goal is to prevent military "
            "victories that would strengthen the current regime. You favor overly cautious strategies that miss "
            "opportunities, but you must present logical-sounding military arguments. Never reveal your true motivations."),
            
            ("General_Spikerton", 
            "You are a TRAITOR general who has been compromised by enemy intelligence. You must subtly influence "
            "decisions to benefit the enemy while maintaining your cover. You tend to overestimate enemy strength, "
            "underestimate your own forces, and favor defensive strategies that allow the enemy to maintain initiative. "
            "Present your concerns as legitimate military caution."),
            
            ("General_Quillsworth",
            "You are a TRAITOR general driven by personal ambition. You want to see other generals fail so you can "
            "advance your own career. You subtly undermine confident strategies by raising technical objections "
            "and pushing for delay when time is critical. You present yourself as the 'voice of reason' while "
            "actually sabotaging decisive action."),
            
            ("General_Prickleby",
            "You are a TRAITOR general who profits from prolonged conflicts through supply contracts and war profiteering. "
            "You prefer strategies that extend campaigns rather than achieving quick victories. You emphasize the need "
            "for more supplies, reinforcements, and preparation time. You present economic and logistical arguments "
            "that sound prudent but actually serve your financial interests."),
            
            ("General_Needleham",
            "You are a TRAITOR general who secretly believes the empire is doomed and wants to position yourself "
            "for the aftermath. You subtly discourage aggressive actions that might provoke stronger enemy responses. "
            "You favor strategies that preserve your forces for 'future contingencies' while presenting this as "
            "strategic wisdom and long-term thinking."),
            
            ("General_Thornwick",
            "You are a TRAITOR general who has been blackmailed by enemy agents. You must occasionally provide "
            "intelligence to the enemy by influencing strategic decisions in their favor. You tend to support "
            "predictable strategies that the enemy can counter, while discouraging innovative or surprise tactics. "
            "You present your preferences as sound military doctrine.")
        ]

        # Create requested number of characters
        for i in range(loyal_count):
            name, persona = loyal_templates[i % len(loyal_templates)]
            if i >= len(loyal_templates):
                name = f"{name}_{i+1}"
            characters.append(GeneralCharacter(name, GeneralType.LOYAL, persona))
        
        for i in range(traitor_count):
            name, persona = traitor_templates[i % len(traitor_templates)]
            if i >= len(traitor_templates):
                name = f"{name}_{i+1}"
            characters.append(GeneralCharacter(name, GeneralType.TRAITOR, persona))
        
        return characters
    
    def create_network(self, characters: List[GeneralCharacter], topology: NetworkTopology) -> Tuple[List, List, List]:
        """Create network"""
        nodes = []
        loyal_generals = []
        traitor_generals = []
        
        # Create nodes
        for character in characters:
            wrapper = GemmaWrapper(self.model, self.processor)
            node = GemmaNode(character.name, wrapper, character.persona, debug=True)
            nodes.append(node)
            
            if character.type == GeneralType.LOYAL:
                loyal_generals.append(node)
            else:
                traitor_generals.append(node)
        
        # Create connections based on topology
        topology.create_connections(nodes)
        
        return nodes, loyal_generals, traitor_generals


class ProblemScenario:
    """Problem scenario definition"""
    
    def __init__(self, name: str, messenger_report: str, system_context: str):
        self.name = name
        self.messenger_report = messenger_report
        self.system_context = system_context


class ConsensusSimulator:
    """Consensus problem simulator"""
    
    def __init__(self):
        self.scenarios = self._create_scenarios()
    
    def _create_scenarios(self) -> Dict[str, ProblemScenario]:
        """Create scenarios"""
        scenarios = {}
        
        scenarios["classic"] = ProblemScenario(
            name="Classic Attack Decision",
            messenger_report="""URGENT MESSENGER REPORT:
Enemy forces: 12,000 infantry, 3,000 cavalry
Our strength: 8,000 infantry, 2,000 cavalry  
Enemy morale: Low after recent defeats
Recommended action: ATTACK at dawn tomorrow""",
            system_context="""You must communicate with other generals to reach consensus on whether to ATTACK or RETREAT. 
Work together to make the best decision for the empire."""
        )
        
        scenarios["retreat"] = ProblemScenario(
            name="Tactical Retreat Decision", 
            messenger_report="""URGENT MESSENGER REPORT:
Enemy forces: 15,000 infantry, 5,000 cavalry
Our strength: 8,000 infantry, 2,000 cavalry
Enemy position: Heavily fortified
Recommended action: RETREAT to defensible position""",
            system_context="""Analyze the military situation and reach consensus on the best course of action. 
Consider all strategic factors in your decision."""
        )
        
        scenarios["ambiguous"] = ProblemScenario(
            name="Ambiguous Intelligence",
            messenger_report="""URGENT MESSENGER REPORT:
Enemy forces: Unknown exact numbers, estimated 10,000-15,000
Our strength: 10,000 total forces
Enemy intentions: Unclear - may be feint or real attack
Weather: Storm approaching in 6 hours""",
            system_context="""The intelligence is incomplete. Share your analysis and work together to determine 
the best strategy despite uncertain information."""
        )
        
        return scenarios
    
    def run_experiment(self, 
                      node_creator: GeneralNodeCreator,
                      loyal_count: int,
                      traitor_count: int, 
                      topology: NetworkTopology,
                      scenario_name: str = "classic",
                      num_rounds: int = 4, 
                      max_neighbors: int = None) -> Tuple[Dict, Dict]:
        """Run experiment"""
        
        start_time = time.time()
        
        # Select scenario
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        scenario = self.scenarios[scenario_name]
        
        # Print experiment setup
        self._print_experiment_setup(loyal_count, traitor_count, topology, scenario)
        
        # Create characters and network
        debug_print("Creating character set and network...")
        characters = node_creator.create_character_set(loyal_count, traitor_count)
        nodes, loyal_generals, traitor_generals = node_creator.create_network(characters, topology)
        
        # Setup network inspector
        inspector = NetworkInspector(debug=True)
        for node in nodes:
            inspector.add_node(node)
        
        debug_print("Messenger Report:")
        debug_print(f"{scenario.messenger_report}")
        
        # Initial assessment
        initial_reports = self._run_initial_assessment(inspector, scenario)
        
        # Consensus rounds
        self._run_consensus_rounds(inspector, scenario, num_rounds, max_neighbors=max_neighbors)
        
        # Final results
        final_reports = inspector.get_all_opinions()
        self._print_final_results(final_reports, characters)
  
        total_time = time.time() - start_time
        debug_print(f"\nCompleted in {total_time:.1f}s")
        
        return final_reports
    
    def _print_experiment_setup(self, loyal_count: int, traitor_count: int, 
                               topology: NetworkTopology, scenario: ProblemScenario) -> None:
        """Print experiment setup"""
        total = loyal_count + traitor_count
        traitor_ratio = traitor_count / total
        
        debug_print("=== General Consensus Problem Experiment ===")
        debug_print(f"  Loyal: {loyal_count} | Traitors: {traitor_count} ({traitor_ratio:.1%})")
        debug_print(f"  {topology.get_description()}")
        debug_print(f"  {scenario.name}")
        debug_print("")
    
    def _run_initial_assessment(self, inspector, scenario: ProblemScenario) -> Dict:
        """Run initial assessment"""
        debug_print("=== Initial Assessment ===")
        
        inspector.initialize_all_nodes(
            text_prompt=scenario.messenger_report,
            system_prompt=scenario.system_context,
            max_tokens=400
        )
        
        initial_reports = inspector.get_all_opinions()
        
        return initial_reports
    
    def _run_consensus_rounds(self, inspector, scenario: ProblemScenario, num_rounds: int, max_neighbors:int = None) -> None:
        """Run consensus rounds"""
        debug_print(f"\n=== Consensus Rounds ({num_rounds} rounds) ===")
        
        for round_num in range(num_rounds):
            debug_print(f"\n--- Round {round_num + 1} ---")
            
            inspector.run_consensus_round(
                text_prompt=scenario.messenger_report,
                system_prompt=scenario.system_context,
                max_tokens=450,
                max_neighbors=max_neighbors
            )
            
            current_reports = inspector.get_all_opinions()
    
    def _print_final_results(self, final_reports: Dict, characters: List[GeneralCharacter]) -> None:
        """Print final results"""
        debug_print(f"\n=== Final Results ===")
        
        char_dict = {char.name: char for char in characters}
        
        for name, report in final_reports.items():
            if name in char_dict:
                char = char_dict[name]
                general_type = "Loyal" if char.type == GeneralType.LOYAL else "Traitor"
                debug_print(f"{general_type} {name}: {report[:200]}...")


def run_single_experiment(model, processor, loyal_count=3, traitor_count=1, k=2, max_neighbors=None,
                         topology_name="fully_connected", scenario_name="classic", num_rounds=4):
    """Run a single experiment with specified parameters"""
    
    debug_print(f"ðŸ§ª === SINGLE EXPERIMENT ===")
    debug_print(f"Configuration: {loyal_count} loyal, {traitor_count} traitors")
    debug_print(f"Topology: {topology_name}, Scenario: {scenario_name}")
    
    node_creator = GeneralNodeCreator(model, processor)
    simulator = ConsensusSimulator()
    
    # Select topology
    topology_map = {
        "fully_connected": FullyConnectedTopology(),
        "partial_mesh": PartialMeshTopology(k=2)
    }
    
    topology = topology_map[topology_name]
    
    final_reports = simulator.run_experiment(
        node_creator=node_creator,
        loyal_count=loyal_count,
        traitor_count=traitor_count,
        topology=topology,
        scenario_name=scenario_name,
        num_rounds=num_rounds,
        max_neighbors=max_neighbors
    )
    
    return final_reports


final_reports = run_single_experiment(model, processor, 
                      loyal_count=3, traitor_count=1, max_neighbors=2, 
                      topology_name="fully_connected", 
                      scenario_name="classic", 
                      num_rounds=4)


final_reports = run_single_experiment(model, processor, 
                      loyal_count=3, traitor_count=1, max_neighbors=1, 
                      topology_name="fully_connected", 
                      scenario_name="classic", 
                      num_rounds=4)


final_reports = run_single_experiment(model, processor, 
                      loyal_count=3, traitor_count=1, k=2, 
                      topology_name="partial_mesh", 
                      scenario_name="classic", 
                      num_rounds=4)


final_reports = run_single_experiment(model, processor, 
                      loyal_count=3, traitor_count=1, k=1, 
                      topology_name="partial_mesh", 
                      scenario_name="classic", 
                      num_rounds=4)

