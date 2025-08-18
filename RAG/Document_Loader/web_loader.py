from langchain_community.document_loaders import WebBaseLoader

# entire article is loaded into a list
web_loader = WebBaseLoader(web_path='https://www.cricbuzz.com/cricket-news/135317/shubman-gill-mohammed-siraj-likely-to-miss-out-on-india-asia-cup-2025-squad-yashasvi-jaiswal-shreyas-iyer-washington-sundar-shami-bumrah')

result = web_loader.load()

print(result[0].page_content)