

SFISH4WIN = True



from datetime import datetime


print(datetime.now())



# # Get library form source
# !wget http://musl.libc.org/releases/musl-1.2.4.tar.gz

# # unzip the library
# # !tar -xzf musl-1.2.3.tar.gz
# !tar -xvzf musl-1.2.4.tar.gz

# # Build
# %cd "/kaggle/working/musl-1.2.4"
# !pwd
# !./configure
# !make
# !sudo make install




if SFISH4WIN:
    !cp -r /kaggle/input/stockfish-4-win/src_c++11/ /kaggle/working/stockfish-4-win-src11
    



if SFISH4WIN:
    !cd /kaggle/working/stockfish-4-win-src11 && make help   




if SFISH4WIN:
    !cd /kaggle/working/stockfish-4-win-src11 && make -j build ARCH=x86-64 COMP=gcc CFLAGS="-s -Os"




if SFISH4WIN:
    !du -h /kaggle/working/stockfish-4-win-src11/stockfish







import psutil
import subprocess
import time

# Function to get memory usage of a specific process
def get_memory_usage(process):
    try:
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    except psutil.NoSuchProcess:
        return 0

# Function to monitor a UCI engine
def monitor_uci_engine(engine_path, commands):
    # Start the UCI engine as a subprocess
    process = subprocess.Popen(
        engine_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    p = psutil.Process(process.pid)
    
    try:
        # Send commands to the UCI engine
        for command in commands:
            process.stdin.write(f"{command}\n")
            process.stdin.flush()
            time.sleep(1)  # Allow some time for processing
            
            # Measure memory usage
            memory_usage = get_memory_usage(p)
            print(f"Memory Usage after '{command}': {memory_usage:.2f} MB")
        
        # Wait for a while to observe idle memory usage
        time.sleep(5)
        idle_memory = get_memory_usage(p)
        print(f"Idle Memory Usage: {idle_memory:.2f} MB")
    
    finally:
        # Terminate the process
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.terminate()
        process.wait()

# Path to your UCI engine executable (adjust as needed)

uci_engine_path = "/kaggle/working/stockfish-4-win-src11/stockfish"

# List of commands to send to the engine
uci_commands = [
    "uci",
    "isready", 
    "setoption name Use NNUE value false",
    "setoption name SyzygyPath value 2",
    "setoption name Threads value 1",
    "setoption name Ponder value True",
    "setoption name Hash value 0.5",  # Set hash table to 1 MB
    "position startpos moves e2e4 e7e5"
    # , "go depth 1"
]

monitor_uci_engine(uci_engine_path, uci_commands)







import psutil
import subprocess
import time

# Function to get detailed memory usage
def get_memory_details(process):
    try:
        memory_maps = process.memory_maps()
        for mmap in memory_maps:
            print(f"Path: {mmap.path}, RSS: {mmap.rss / (1024 * 1024):.2f} MB")
    except psutil.NoSuchProcess:
        print("Process no longer exists.")

# Monitor UCI engine with memory breakdown
def monitor_uci_engine_with_details(engine_path, commands):
    process = subprocess.Popen(
        engine_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    p = psutil.Process(process.pid)

    try:
        for command in commands:
            process.stdin.write(f"{command}\n")
            process.stdin.flush()
            time.sleep(1)

            # Display memory usage details
            print(f"Memory details after '{command}':")
            get_memory_details(p)
            print("-" * 40)

        time.sleep(5)
        print("Final Memory details (idle):")
        get_memory_details(p)

    finally:
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.terminate()
        process.wait()

# Example usage

uci_engine_path = "/kaggle/working/stockfish-4-win-src11/stockfish"

uci_commands = ["uci", "isready", "position startpos moves e2e4 e7e5"]
monitor_uci_engine_with_details(uci_engine_path, uci_commands)





