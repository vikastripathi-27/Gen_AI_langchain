from ddgs import DDGS

with DDGS() as ddgs:
    # Text search
    results = ddgs.text("what is artifical intelligence", max_results=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['href']}")
        print(f"   Snippet: {result['body']}")
        
        print("-" * 50)

    # News search
    news_results = ddgs.news("what is india's squad for asia cup", max_results=3)
    for i, result in enumerate(news_results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Source: {result['source']}")
        print(f"   Date: {result['date']}")
        print(f"   URL: {result['url']}")

        print("-" * 50)
        
    # Image search
    image_results = ddgs.images("ms dhoni", max_results=3)
    for i, result in enumerate(image_results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Image URL: {result['image']}")
        print(f"   Source: {result['url']}")