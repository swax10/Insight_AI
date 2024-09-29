import requests
from pathlib import Path
import re
from tools.get_research_papers import fetch_arxiv_papers, extract_paper_info

def download_paper(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")

def sanitize_filename(title):
    # Remove or replace invalid characters
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with underscores
    title = title.replace(' ', '_')
    # Limit the length to 50 characters
    return title[:50]

def download_research_papers(search_topic, output_dir):
    # Create research_papers folder if it doesn't exist
    research_papers_dir = Path(output_dir)
    research_papers_dir.mkdir(exist_ok=True)

    # Fetch and download papers
    arxiv_response = fetch_arxiv_papers(search_topic)
    paper_info_list, _ = extract_paper_info(arxiv_response)

    for paper in paper_info_list[:1]:
        safe_title = sanitize_filename(paper['title'])
        filename = research_papers_dir / f"{safe_title}.pdf"
        download_paper(paper['pdf_url'], str(filename))

    return str(research_papers_dir)
