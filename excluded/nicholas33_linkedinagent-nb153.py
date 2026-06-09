# SECTION A: Imports & Environment Setup
import os
import re
import json
import time
import random
import requests
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Install required packages
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
packages = [
    "trafilatura",
    "readability-lxml",
    "beautifulsoup4",
    "requests",
    "python-dotenv",
    "groq"
]

for package in packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"âœ… {package} already installed")
    except ImportError:
        print(f"ğŸ“¦ Installing {package}...")
        install_package(package)

# Import all required libraries
import trafilatura
from bs4 import BeautifulSoup
from groq import Groq
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("âœ… All dependencies loaded successfully!")


# SECTION B: Helper Functions

def is_valid_url(url: str) -> bool:
    """Validate URL format and accessibility"""
    try:
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Parse URL
        parsed = urlparse(url)

        # Check if domain is valid
        if not parsed.netloc or '.' not in parsed.netloc:
            return False

        return True
    except:
        return False

def download_html(url: str, max_retries: int = 5) -> str:
    """Download HTML content from URL with aggressive rate limiting and retry logic"""
    import random
    import time
    
    # Enhanced user agents for better stealth
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]

    # Create a session to maintain cookies across retries
    session = requests.Session()
    
    # Enhanced headers to bypass anti-bot measures
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'Referer': 'https://www.google.com/'  # Makes it look like we came from Google
    }

    session.headers.update(headers)

    for attempt in range(max_retries + 1):
        try:
            # Only add delays on retry attempts, not the first attempt
            if attempt > 0:
                # Retry attempts: exponential backoff with jitter
                base_delay = 2 ** attempt  # 2, 4, 8 seconds
                jitter = random.uniform(0.5, 1.5)
                delay = base_delay * jitter
                logger.info(f"ğŸ”„ Rate limited (429). Retrying in {delay:.1f} seconds... (attempt {attempt}/{max_retries})")
                time.sleep(delay)

            logger.info(f"ğŸ“¥ Downloading HTML from: {url}")
            response = session.get(url, timeout=30, allow_redirects=True)

            # Handle different response codes
            if response.status_code == 429:
                if attempt == max_retries:
                    raise Exception(f"Rate limited after {max_retries} retries: {response.status_code}")
                # Rotate user agent on 429 errors
                new_user_agent = random.choice(user_agents)
                session.headers.update({'User-Agent': new_user_agent})
                logger.info(f"ğŸ”„ Rotating User-Agent due to 429 error")
                continue  # Retry after delay
            elif response.status_code == 403:
                # Forbidden - might be blocked, try different user agent
                logger.warning("âš ï¸� 403 Forbidden - rotating user agent...")
                new_user_agent = random.choice(user_agents)
                session.headers.update({'User-Agent': new_user_agent})
                if attempt == max_retries:
                    response.raise_for_status()
                continue
            else:
                response.raise_for_status()

            return response.text

        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                logger.error(f"â�Œ Failed to download HTML after {max_retries} attempts: {e}")
                raise Exception(f"Failed to download webpage after retries: {e}")
            else:
                logger.warning(f"âš ï¸� Request failed (attempt {attempt + 1}): {e}")
                continue

    # This should never be reached, but just in case
    raise Exception("Unexpected error in download function")

def extract_text(html: str) -> str:
    """Extract readable text from HTML using trafilatura with fallback to BeautifulSoup"""
    try:
        logger.info("ğŸ“� Extracting text content...")

        # Try trafilatura first (best for article extraction)
        text = trafilatura.extract(html,
                                 include_comments=False,
                                 include_tables=True,
                                 prune_xpath=['//div[@class="advertisement"]', '//script', '//style'])

        if text and len(text.strip()) > 100:
            # Clean and limit text
            text = re.sub(r'\s+', ' ', text.strip())
            return text[:10000]  # Limit to 10k characters

        # Fallback to BeautifulSoup
        logger.info("ğŸ”„ Using BeautifulSoup fallback...")
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()

        # Extract text from main content areas
        content_selectors = ['article', 'main', '.content', '.post', '.entry', '#content', '#main']
        text = ""
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                text = content.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    break

        if not text:
            # Last resort: get all paragraph text
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Clean and limit text
        text = re.sub(r'\s+', ' ', text.strip())
        return text[:10000] if text else ""

    except Exception as e:
        logger.error(f"â�Œ Text extraction failed: {e}")
        return ""


# SECTION C: LLM Integration

# Load Groq API key from Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GROQ_API_KEY = user_secrets.get_secret("GROQ_API_KEY")
    print("âœ… Groq API key loaded from Kaggle secrets!")
except Exception as e:
    print(f"âš ï¸� Could not load from Kaggle secrets: {e}")
    # Fallback to environment variable or manual input
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    if not GROQ_API_KEY:
        GROQ_API_KEY = input("ğŸ”‘ Enter your Groq API Key: ").strip()

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)
print("âœ… Groq client initialized!")

def generate_linkedin_post(article_text: str, original_url: str) -> str:
    """Generate LinkedIn-optimized post directly from article content"""
    prompt = f"""- Write a LinkedIn blog post (300 words) summarizing and analyzing the key points from the following article: {original_url}. Focus on the main announcement, its significance for the industry, notable features or innovations, pricing or market positioning if available, and any interesting tech facts. Highlight the broader strategic implications and end with a question to engage LinkedIn professionals and invite discussion.

    Article Content: {article_text[:8000]}

    Requirements:
    - Start with an appropriate emoji and cool catchy title
    - NO markdown formatting (no **bold**, no *italics*, no headers with #, no double asterisk)
    - 3-4 sentence business summary (more detailed than usual)
    - 2-3 key insights takeaways (summarize each point)
    - Short conclusion paragraph (2-3 sentences) after insights
    - 200-300 words total (aim for comprehensive coverage)
    - End with 3-5 relevant hashtags
    - Include a call-to-action with the original article link
    - Tech, Cool and approachable style
    - NO markdown formatting (no **bold**, no *italics*, no headers with #)
    - CRITICAL: Use blank lines to separate sections for readability

    Format structure (MUST include blank lines between sections):
    ğŸš€ Cool Catchy Title

    ğŸ“š [3-4 sentence detailed business summary]

    âœ¨ Key insights:
    â€¢ [Insight 1]
    â€¢ [Insight 2]
    â€¢ [Insight 3]

    ğŸ“£ [2-3 sentence conclusion wrapping up the main takeaway]

    ğŸ‘‰ Read the full article here: {original_url}

    #hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5

    IMPORTANT: Ensure there are blank lines between each section (title, summary, insights, conclusion, link, hashtags) for readability.

    Write the complete LinkedIn post:"""

    try:
        logger.info("âœ�ï¸� Generating LinkedIn post...")
        response = groq_client.chat.completions.create(
            # model="llama-3.1-8b-instant",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"â�Œ Post generation failed: {e}")
        return f"ğŸš€ Exciting tech insights discovered! Check out this interesting content about emerging technologies and innovation. #tech #innovation #futureofwork"


# SECTION D: LinkedIn API Integration (UPDATED - Using linkedin_posting.py approach)

# Load LinkedIn access token from Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    LINKEDIN_ACCESS_TOKEN = user_secrets.get_secret("LINKEDIN_ACCESS_TOKEN")
    print("âœ… LinkedIn access token loaded from Kaggle secrets!")
except Exception as e:
    print(f"âš ï¸� Could not load LinkedIn token from Kaggle secrets: {e}")
    # Fallback to environment variable or manual input
    LINKEDIN_ACCESS_TOKEN = os.getenv('LINKEDIN_ACCESS_TOKEN')
    if not LINKEDIN_ACCESS_TOKEN:
        LINKEDIN_ACCESS_TOKEN = input("ğŸ”‘ Enter your LinkedIn Access Token: ").strip()

# ============================================
# GET YOUR LINKEDIN PROFILE INFO (Auto-detection from linkedin_posting.py)
# ============================================
print("ğŸ‘¤ Getting your LinkedIn profile info...")

userinfo_response = requests.get(
    "https://api.linkedin.com/v2/userinfo",
    headers={"Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}"}
)

if userinfo_response.status_code != 200:
    print("â�Œ Failed to get profile info!")
    print("Error:", userinfo_response.text)
    print("\nğŸ”„ Your token may have expired. Get a new one from:")
    print("https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id=78migzqtiqpx3v&redirect_uri=http://localhost:8000/callback&scope=openid%20profile%20w_member_social")
    raise Exception("LinkedIn token invalid or expired")

userinfo = userinfo_response.json()
person_id = userinfo['sub']
LINKEDIN_USER_URN = f"urn:li:person:{person_id}"

print(f"âœ… Auto-detected LinkedIn User URN: {LINKEDIN_USER_URN}")

def publish_to_linkedin(post_text: str) -> str:
    """Publish post to LinkedIn using linkedin_posting.py approach"""
    try:
        print("ğŸ“� Publishing to LinkedIn...")

        # LinkedIn UGC Post API payload (from linkedin_posting.py)
        post_payload = {
            "author": LINKEDIN_USER_URN,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": post_text
                    },
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }

        headers = {
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }

        response = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers=headers,
            json=post_payload,
            timeout=30
        )

        if response.status_code == 201:
            # Extract post ID from response headers (linkedin_posting.py style)
            post_id = response.headers.get('X-RestLi-Id', '')
            if not post_id:
                # Fallback: try to get from response body
                post_data = response.json()
                post_id = post_data.get('id', '')

            # Convert to public URL
            post_url = f"https://www.linkedin.com/feed/update/{post_id}" if post_id else "https://www.linkedin.com/feed/"
            print(f"âœ… Successfully published to LinkedIn!")
            print(f"Post ID: {post_id}")
            print(f"View your post at: {post_url}")
            return post_url
        else:
            print(f"â�Œ Failed to post. Status: {response.status_code}")
            print("Response:", response.text)
            raise Exception(f"LinkedIn API error: {response.status_code}")

    except Exception as e:
        print(f"â�Œ LinkedIn publishing failed: {e}")
        raise Exception(f"Failed to publish to LinkedIn: {e}")

# Test LinkedIn connection
print("âœ… LinkedIn API integration ready!")
print(f"ğŸ“� Will post as: {LINKEDIN_USER_URN}")


# SECTION E: Agent Implementations

def url_ingestion_agent(url: str) -> str:
    """Agent 1: URL Ingestion - validates URL, downloads HTML, extracts text"""
    try:
        logger.info(f"ğŸ”— Starting URL ingestion for: {url}")

        # Step 1: Validate URL
        if not is_valid_url(url):
            raise Exception(f"Invalid URL format: {url}")

        # Step 2: Download HTML
        html = download_html(url)
        if not html or len(html.strip()) < 100:
            raise Exception("Downloaded content is too short or empty")

        # Step 3: Extract readable text
        text = extract_text(html)
        if not text or len(text.strip()) < 50:
            raise Exception("Could not extract sufficient readable text from the webpage")

        logger.info(f"âœ… Successfully extracted {len(text)} characters of content")
        return text

    except Exception as e:
        logger.error(f"â�Œ URL ingestion failed: {e}")
        raise Exception(f"Failed to process URL {url}: {e}")


def post_writer_agent(article_text: str, original_url: str) -> str:
    """Agent 2: Post Writer - generates LinkedIn-optimized post directly from content"""
    try:
        logger.info("âœ�ï¸� Crafting LinkedIn post...")

        # Generate LinkedIn post directly from article content
        post_text = generate_linkedin_post(article_text, original_url)

        if not post_text or len(post_text.strip()) < 10:
            raise Exception("Generated post is too short or empty")

        # Basic validation - check word count (under 500 words to allow for longer posts)
        word_count = len(post_text.split())
        if word_count > 500:
            logger.warning(f"Post word count ({word_count}) exceeds 500 word limit, truncating...")
            words = post_text.split()[:495]  # Keep ~495 words to leave room for "..."
            post_text = ' '.join(words) + "..."

        logger.info(f"âœ… Generated post ({word_count} words) with {post_text.count('#')} hashtags")
        return post_text

    except Exception as e:
        logger.error(f"â�Œ Post generation failed: {e}")
        raise Exception(f"Failed to generate LinkedIn post: {e}")

def publisher_agent(post_text: str) -> str:
    """Agent 3: Publisher - publishes to LinkedIn and returns URL"""
    try:
        logger.info("ğŸš€ Publishing to LinkedIn...")

        # Publish to LinkedIn
        post_url = publish_to_linkedin(post_text)

        if not post_url:
            raise Exception("LinkedIn API returned empty post URL")

        logger.info(f"âœ… Successfully published! Post URL: {post_url}")
        return post_url

    except Exception as e:
        logger.error(f"â�Œ Publishing failed: {e}")
        raise Exception(f"Failed to publish to LinkedIn: {e}")

# Main Orchestration Function
def run_agent_system(url: str) -> str:
    """Main function that orchestrates the complete LinkedIn automation pipeline"""
    try:
        print("\nğŸ¤– Starting LinkedIn Automation Agent...")
        print("=" * 50)

        # Agent 1: URL Ingestion
        print("ğŸ“¥ Step 1/4: Ingesting content from URL...")
        article_text = url_ingestion_agent(url)

        # Agent 2: Post Writing (Direct from content)
        print("âœ�ï¸� Step 2/4: Generating LinkedIn post...")
        linkedin_post = post_writer_agent(article_text, url)

        # Agent 3: Preview & Confirmation
        print("ğŸ”� Step 3/4: Preview and confirmation...")

        # ğŸ”� NEW: Display generated post for review
        print("\n" + "=" * 60)
        print("ğŸ“� GENERATED LINKEDIN POST (PREVIEW)")
        print("=" * 60)
        print(linkedin_post)
        print("=" * 60)

        # Ask for confirmation before publishing
        while True:
            confirmation = input("\nğŸš€ Do you want to publish this post? (y/n/edit/r): ").strip().lower()

            if confirmation in ['y', 'yes']:
                print("\nâœ… Proceeding with publication...")
                break
            elif confirmation in ['n', 'no']:
                print("\nâš ï¸� Publication cancelled by user.")
                return "cancelled"
            elif confirmation in ['edit', 'e']:
                print("\nâœ�ï¸� Enter your edited version of the post (multiline):")
                print("Type your edited post below. When finished, type 'END' on a new line.")
                print("=" * 60)

                edited_lines = []
                while True:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    edited_lines.append(line)

                edited_post = '\n'.join(edited_lines).strip()
                
                if edited_post:
                    linkedin_post = edited_post
                    print("\nğŸ“� Updated post:")
                    print("=" * 60)
                    print(linkedin_post)
                    print("=" * 60)
                    continue
                else:
                    print("\nâš ï¸� Edit cancelled. Publication cancelled by user.")
                    return "cancelled"
            elif confirmation in ['r', 'restart']:
                print("\nğŸ”„ Restarting... Let's try a different article!")
                print("\n" + "=" * 60)
                return "restart"
            else:
                print("Please enter 'y' (yes), 'n' (no), 'edit' (to modify the post), or 'r' (restart with new URL)")

        # Agent 3: Publishing (only if confirmed)
        print("\nğŸš€ Step 4/4: Publishing to LinkedIn...")
        post_url = publisher_agent(linkedin_post)

        print("\nâœ… SUCCESS! LinkedIn automation complete!")
        print("=" * 50)

        return post_url

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        print(f"\nâ�Œ {error_msg}")
        logger.error(error_msg)
        raise Exception(error_msg)

print("âœ… Agent system ready!")


# SECTION F: User Interface & Main Execution

def main():
    """Main user interface for the LinkedIn Automation Agent"""
    print("\nğŸš€ LinkedIn Automation Agent - Content Generation & Repurposing")
    print("=" * 60)
    print("This agent will:")
    print("1. ğŸ“¥ Ingest content from any article URL")
    print("2. âœ�ï¸� Generate a LinkedIn-optimized post")
    print("3. ğŸ”� Preview and confirm the post")
    print("4. ğŸš€ Publish to LinkedIn (with your approval)")
    print("=" * 60)

    while True:  # Main loop for restart functionality
        try:
            # Get URL from user
            input_url = input("\nğŸ”— Enter the article URL to convert to LinkedIn post: ").strip()

            if not input_url:
                print("â�Œ No URL provided. Exiting.")
                break

            # Validate URL format
            if not is_valid_url(input_url):
                print(f"â�Œ Invalid URL format: {input_url}")
                continue  # Ask for URL again

            # Run the complete automation pipeline
            post_url = run_agent_system(input_url)

            # Check return values
            if post_url == "cancelled":
                print("\nâ„¹ï¸� Process completed without publishing.")
                continue  # Ask for new URL
            elif post_url == "restart":
                print("\nğŸ”„ Restarting the process...")
                continue  # Restart the main loop to ask for new URL

            # Success message - only reached if post was published
            print("\nğŸ�‰ SUCCESS! Your LinkedIn post has been published!")
            print(f"\nğŸ“± Click here to view your post: {post_url}")
            print("\nğŸ’¡ Tip: The post will appear in your LinkedIn feed shortly.")
            break  # Exit after successful publication

        except KeyboardInterrupt:
            print("\n\nâš ï¸� Operation cancelled by user.")
            break
        except Exception as e:
            print(f"\nâ�Œ An error occurred: {str(e)}")
            print("\nğŸ”§ Troubleshooting tips:")
            print("â€¢ Check your internet connection")
            print("â€¢ Verify your Groq API key is valid")
            print("â€¢ Ensure your LinkedIn access token is current")
            print("â€¢ Make sure the URL is accessible and contains readable content")

# Run the main interface
if __name__ == "__main__":
    main()




