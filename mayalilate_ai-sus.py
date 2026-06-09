!pip install -U -q "google"
!pip install -U -q "google.genai"

import os
from google.colab import userdata
from google.colab import drive
os.environ["GEMINI_API_KEY"] = userdata.get("GOOGLE_API_KEY")

drive.mount("/content/drive")
# Please ensure that uploaded files are available in the AI Studio folder or change the working folder.
os.chdir("/content/drive/MyDrive/Google AI Studio")


# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-lite-preview-06-17"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="audio/wav",
                    data=base64.b64decode(
                        """<Drive file: 1wUQkllPpblAriwwJyzX7u4H0jGO1WMYu>"""
                    ),
                ),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Understanding Complementarity**

I've been exploring the seemingly contradictory names of Allah, moving past concepts of mercy, power, and justice. The current focus is on how seemingly opposing attributes function as complementary facets of the Divine. I'm aiming for a clearer understanding of how these different names coalesce.


**Delving Deeper into Names**

I'm now delving into the specific names, particularly Al-Qabid, Al-Basit, Al-Muizz, and Al-Muzill, seeking clarity on their interconnectedness. My research has begun, and I'm uncovering their meanings and implications. Next, I'll synthesize this information to answer the user's questions about these contrasting pairs, focusing on the influence of names like Al-Lateef and Al-Aleem.


**Examining Contradictions**

I've explored the user's inquiry, focusing on the seeming contradictions in names like Al-Qabid/Al-Basit and Al-Muizz/Al-Muzill. I've compiled search results that shed light on these divine attributes. Now, I'm analyzing the impact of reflecting on Al-Lateef and Al-Aleem within this framework.


**Unpacking Nuances Further**

I'm now diving deeper into the nuances of Al-Qabid/Al-Basit, viewing them as complementary facets of divine control rather than conflicting traits. The results emphasize that Al-Qabid tests and purifies, while Al-Basit offers ease and expansion, creating a necessary balance. I'm focusing on their intertwined nature to avoid the misconception of divine miserliness.



**Addressing the Divine Names**

I'm synthesizing insights on the paradoxical names: Al-Qabid/Al-Basit and Al-Muizz/Al-Muzill. The balance between withholding and expanding, honoring and humbling, are central themes. I'm focusing on their nuanced relationship, illustrating that these seemingly contradictory attributes demonstrate divine control and justice, not inconsistency. The results emphasize that Al-Lateef and Al-Aleem are key to understanding the profound nature of Allah's attributes.


**Reflecting and Synthesizing**

I've organized the information and established a clear framework to address the user's questions. Currently, I am structuring my response around the balance between divine attributes like withholding and expanding. I'm focusing on weaving explanations of Al-Qabid/Al-Basit and Al-Muizz/Al-Muzill into a coherent understanding of Allah's attributes. Now, I will highlight the profound implications of reflecting on Al-Lateef and Al-Aleem.


"""),
                types.Part.from_text(text="""The 99 names of Allah, Asma ul Husna, indeed reveal profound and multifaceted aspects of His divine nature. The pairs of names you mentioned, which might seem contradictory at first glance, actually illustrate the perfect balance and comprehensive wisdom in Allah's attributes.

**Al-Qabid (The Constrictor/Withholder) and Al-Basit (The Expander/Extender)**
These two names are considered inseparable, as one describes a divine action that complements the other. Al-Qabid is the one who withholds, constricts, or seizes, whether it be sustenance, opportunities, or even the souls of people at their appointed time. This act is not punitive but is often a test or a means of refining believers, redirecting them towards spiritual pursuits. Al-Basit, conversely, is the one who expands, extends, and bestows abundance, granting ease, provision, and opening the hearts of His servants to faith and blessings. Together, they represent Allah's control over the ebb and flow of life, a meticulous balance that shapes the believer's journey, often leading to greater knowledge and wisdom. Reflecting on these names encourages a balance between fear of Allah's retribution and hope in His mercy, recognizing that both constriction and expansion serve a divine purpose.

**Al-Muizz (The Honorer/Bestower of Might) and Al-Muzill (The Humiliator/Degrader)**
These names highlight Allah's absolute power to grant honor or to humble whomever He wills. Al-Muizz bestows might, strength, and honor upon His obedient servants, elevating them in dignity and making them respected. This honor comes from Allah and is independent of worldly status. Al-Muzill, on the other hand, is the one who humbles or degrades. This act is a reflection of divine justice, often applied to those who disobey or rebel against Allah, justly abasing them. The key understanding here is that true honor comes from Allah, and perceived humiliation can be a consequence of turning away from Him. This pair emphasizes that Allah is the ultimate source of power and status, and He bestows them according to His wisdom and justice.

**The Impact of Reflecting on Al-Lateef (The Subtle/Kind) and Al-Aleem (The All-Knowing)**
Reflecting on **Al-Lateef** can deeply impact one's inner state and actions. Al-Lateef is the One who is incredibly kind, gentle, and subtle in His dealings with His servants, often providing benefits and solutions in ways that are not perceived. He is fully aware of our situations, our innermost thoughts, and desires. Contemplating Al-Lateef encourages believers to emulate this attribute by being gentle, kind, and considerate towards others, especially the vulnerable. It also fosters a sense of trust, as one understands that Allah is working things out for their ultimate good, even when circumstances are difficult.

Reflecting on **Al-Aleem** (The All-Knowing) reinforces the understanding of Allah's complete and encompassing knowledge. He knows all things, seen and unseen, past, present, and future, with perfect detail. This realization fosters accountability, encouraging sincerity in one's actions and intentions, as nothing is hidden from Allah. It also leads to a deeper reliance on Him, knowing that He is aware of every need and situation. By understanding that Allah is both Al-Lateef and Al-Aleem, a believer recognizes that Allah is not only aware of every subtle detail of their life but also acts with perfect kindness and wisdom in accordance with that knowledge. This dual understanding can inspire a more conscious and reverent approach to life, seeking Allah's guidance and accepting His decree with peace."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    tools = [
        types.Tool(url_context=types.UrlContext()),
        types.Tool(googleSearch=types.GoogleSearch(
        )),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=8192,
        ),
        tools=tools,
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()


