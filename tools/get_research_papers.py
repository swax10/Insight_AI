import requests
import time
import lxml.etree as lxml_etree

def fetch_arxiv_papers(topic, max_results=10):
    response = requests.get(f"https://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}")
    return response.text

def extract_paper_info(xml_content):
    start_time = time.perf_counter()
    root = lxml_etree.fromstring(xml_content.encode('utf-8'))
    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
    
    paper_list = []
    for entry in entries:
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        links = entry.findall('{http://www.w3.org/2005/Atom}link')
        pdf_url = next((link.get('href') for link in links if link.get('title') == 'pdf'), None)
        
        if title and pdf_url:
            paper_list.append({
                'title': title,
                'pdf_url': pdf_url
            })
    
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    return paper_list, processing_time

if __name__ == "__main__":
    search_topic = input("Enter the research topic: ")
    arxiv_response = fetch_arxiv_papers(search_topic)

    paper_info_list, extraction_time = extract_paper_info(arxiv_response)

    print(f"Information extraction time: {extraction_time:.6f} seconds")
    print(f"Number of papers found: {len(paper_info_list)}")
    
    for paper_info in paper_info_list:
        print(f"Title: {paper_info['title']}")
        print(f"PDF URL: {paper_info['pdf_url']}")
        print("---")