from crewai_web_search import WebSearchCrew
import os


def main():
    print("CrewAI Web Search")
    
    # Check API keys
    has_serper = bool(os.getenv("SERPER_API_KEY"))
    has_firecrawl = bool(os.getenv("FIRECRAWL_API_KEY"))
    
    if not (has_serper and has_firecrawl):
        print("⚠️  API Keys not configured!")
        print("\nTo use web search functionality:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys:")
        print("   - SERPER_API_KEY from https://serpapi.com/")
        print("   - FIRECRAWL_API_KEY from https://firecrawl.dev/")
        print("\n Running in demo mode (will show errors)...\n")
    
    # Initialize crew
    crew = WebSearchCrew()
    
    # Example queries that require web search and crawling
    queries = [
        "What are the latest features in Python 3.14?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}")
        print(f"{'='*70}")
        
        try:
            result = crew.search(query, verbose=True)
            
            print(f"\n{'='*70}")
            print("FINAL RESULT:")
            print(f"{'='*70}")
            print(result)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nNote: This likely means API keys are not configured.")
            print("The system will work perfectly once you set up:")
            print("  - SERPER_API_KEY")
            print("  - FIRECRAWL_API_KEY")
            break
        
        if i < len(queries):
            print("\n" + "="*70)
            input("Press Enter to continue to next example...")


if __name__ == "__main__":
    main()
