"""
Efficient CrewAI-Based Web Search System
========================================

Architecture for minimal API usage:
1. Web Searcher - Uses SerpAPI only when needed
2. Content Crawler - Uses Firecrawl only for essential pages
3. Synthesizer - Combines and presents results

USAGE:
    from crewai_web_search import WebSearchCrew
    
    crew = WebSearchCrew(
        serper_api_key="your-key",
        firecrawl_api_key="your-key"
    )
    result = crew.search("What are the latest AI breakthroughs?")
    print(result)
"""

import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import json

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool  # type: ignore

# LLM imports
from langchain_openai import ChatOpenAI  # type: ignore
from langchain_community.llms import Ollama  # type: ignore

# Web search tools
from serpapi import GoogleSearch  # type: ignore
from firecrawl import FirecrawlApp  # type: ignore


# Load environment variables
load_dotenv()


class WebSearchCrew:
    """Efficient web search crew with minimal API usage"""
    
    def __init__(
        self,
        serper_api_key: Optional[str] = None,
        firecrawl_api_key: Optional[str] = None,
        model_name: str = "gemma3n:e2b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the web search crew
        
        Args:
            serper_api_key: SerpAPI key (or set SERPER_API_KEY env var)
            firecrawl_api_key: Firecrawl key (or set FIRECRAWL_API_KEY env var)
            model_name: Ollama model name
            ollama_base_url: Ollama server URL
        """
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        self.model_name = model_name
        
        # Initialize LLM - CrewAI uses litellm which needs provider prefix
        # For Ollama, we need to use "ollama/<model_name>" format
        self.llm = ChatOpenAI(
            base_url=f"{ollama_base_url}/v1",
            api_key="ollama",  # Dummy key for Ollama
            model=f"ollama/{model_name}",  # Add ollama/ prefix for litellm
            temperature=0.7,
        )
        
        # Cache for search results to avoid duplicate calls
        self.search_cache: Dict[str, Any] = {}
        self.crawl_cache: Dict[str, str] = {}
        
        # Initialize clients
        self.firecrawl = None
        if self.firecrawl_api_key:
            try:
                self.firecrawl = FirecrawlApp(api_key=self.firecrawl_api_key)
            except Exception as e:
                print(f"⚠️  Firecrawl initialization warning: {e}")
        
    def _create_serper_tool(self):
        """Create optimized SerpAPI search tool"""
        
        @tool("web_search")
        def web_search(query: str) -> str:
            """
            Search the web using SerpAPI. Returns top search results with URLs and snippets.
            You MUST use the returned URLs with the scrape_webpage tool to get actual content.
            
            Args:
                query: The search query
                
            Returns:
                Formatted string with search results including URLs for crawling
            """
            # Check cache first
            if query in self.search_cache:
                print(f"📦 Using cached search results for: {query}")
                cached = self.search_cache[query]
                return self._format_search_results(cached)
            
            if not self.serper_api_key:
                return "ERROR: SERPER_API_KEY not provided. Set SERPER_API_KEY environment variable or pass to constructor."
            
            try:
                print(f"🔍 Searching web for: {query}")
                search = GoogleSearch({
                    "q": query,
                    "api_key": self.serper_api_key,
                    "num": 3  # Get 3 results
                })
                results = search.get_dict()
                
                # Extract organic results with full details
                organic_results = results.get("organic_results", [])[:5]
                
                # Store in cache
                self.search_cache[query] = organic_results
                
                return self._format_search_results(organic_results)
                
            except Exception as e:
                return f"ERROR: Search failed - {str(e)}"
        
        return web_search
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results in a clear, structured way"""
        if not results:
            return "No search results found."
        
        output = ["SEARCH RESULTS (You MUST crawl the top 2 URLs to get accurate information):", ""]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("link", "")
            snippet = result.get("snippet", "No snippet")
            
            output.append(f"[{i}] {title}")
            output.append(f"    URL: {url}")
            output.append(f"    Snippet: {snippet}")
            output.append("")
        
        output.append("IMPORTANT: Snippets alone are insufficient. You MUST use scrape_webpage on the top 2 URLs.")
        return "\n".join(output)
    
    def _create_crawler_tool(self):
        """Create optimized Firecrawl scraping tool"""
        
        @tool("scrape_webpage")
        def scrape_webpage(url: str) -> str:
            """
            Scrape detailed content from a specific webpage using Firecrawl.
            This tool extracts the actual content from web pages.
            Try to crawl URLs one by one - if one fails, move to the next.
            
            Args:
                url: The complete URL to scrape (must start with http:// or https://)
                
            Returns:
                Extracted text content from the page with source citation, or ERROR message if failed
            """
            # Clean the URL - strip whitespace and trailing slashes
            url = url.strip().rstrip('/')
            
            # Check cache first
            if url in self.crawl_cache:
                print(f"Using cached content for: {url}")
                cached_content = self.crawl_cache[url]
                return f"SUCCESS: Crawled from cache\nSOURCE: {url}\n\n{cached_content}"
            
            if not self.firecrawl:
                return f"CRAWL_FAILED: Firecrawl not available. API key not configured."
            
            try:
                print(f"🕷️  Crawling: {url}")
                result = self.firecrawl.scrape(  # type: ignore
                    url,
                    formats=['markdown'],
                    only_main_content=True  # Get only main content
                )
                
                # Firecrawl v2 returns a Document object with markdown attribute
                content = result.markdown[:8000] if result.markdown else ""  # Limit to 8000 chars
                
                if not content or len(content.strip()) < 50:
                    error_msg = f"CRAWL_FAILED: Minimal/no content extracted from {url}"
                    print(error_msg)
                    return error_msg
                
                # Format with source citation and success marker
                formatted = f"SUCCESS: Content crawled successfully\nSOURCE: {url}\n\n{content}"
                
                # Cache the result (without prefix to avoid duplication)
                self.crawl_cache[url] = content
                
                print(f"Successfully crawled {url} ({len(content)} chars)")
                return formatted
                
            except Exception as e:
                error_msg = f"❌ CRAWL_FAILED: {url} - {str(e)}"
                print(error_msg)
                return error_msg
        
        return scrape_webpage
    
    def create_agents(self):
        """Create the crew agents with clear, strict roles"""
        
        # Tools
        web_search_tool = self._create_serper_tool()
        scrape_tool = self._create_crawler_tool()
        
        # 1. Web Searcher Agent - Executes web searches ONLY
        searcher = Agent(
            role="Web Search Specialist",
            goal="Execute ONE precise web search and extract URLs for crawling",
            backstory="""You are a web search expert. Your ONLY job is to:
            1. Take the user's query
            2. Use the web_search tool ONCE with an optimized search query
            3. Return the search results showing the URLs found
            
            You do NOT answer questions. You do NOT interpret results. 
            You ONLY search and report what URLs were found.""",
            tools=[web_search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
        
        # 2. Content Crawler Agent - Scrapes the top 2 URLs
        crawler = Agent(
            role="Web Content Crawler",
            goal="Try to crawl top 2 URLs from search results; if both fail, use snippets as fallback",
            backstory="""You are a pragmatic web scraping expert. Your approach:
            
            1. Try to crawl the first URL from search results
               - If successful (SUCCESS) → excellent, you have content
               - If failed (CRAWL_FAILED) → no problem, try the next one
            
            2. Try to crawl the second URL from search results
               - If successful → great, now you have content from 1 or 2 sources
               - If failed → that's okay too
            
            3. Fallback strategy:
               - If at least one URL succeeded → use that crawled content
               - If both URLs failed → use the snippets from search results instead
            
            CRITICAL RULES:
            - Try each URL only ONCE - never retry the same URL
            - Copy URLs exactly as shown in search results
            - Don't modify URLs (don't add/remove trailing slashes, etc.)
            - Move on quickly if a URL fails - don't get stuck
            - If both crawls fail, snippets are acceptable (better than nothing!)
            
            You are efficient and pragmatic - you don't waste time retrying failed operations.""",
            tools=[scrape_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2,  # Limit iterations to prevent infinite loops
        )
        
        # 3. Synthesizer Agent - Combines crawled information or snippets
        synthesizer = Agent(
            role="Information Synthesizer",
            goal="Create accurate answers using crawled content if available, or snippets if crawling failed",
            backstory="""You are an expert at synthesizing information from web sources.
            
            CRITICAL RULES:
            1. Check what content is available:
               - If you have crawled content (marked with "SUCCESS" and "SOURCE:") → Use that!
               - If all crawls failed (marked with "CRAWL_FAILED") → Use snippets from search results
            
            2. Use ONLY the information provided to you:
               - Do NOT make up or hallucinate any information
               - Do NOT add information from your training data
               - Stick to what's in the crawled content or snippets
            
            3. If crawled content is available:
               - Use the detailed information from crawled pages
               - Cite the SOURCE URLs
            
            4. If you only have snippets (because crawling failed):
               - Use the snippet information from search results
               - State clearly: "Based on search snippets (full page crawling was unavailable)"
               - Still cite the URLs where snippets came from
            
            FORMAT YOUR RESPONSE AS:

            **Answer:**
            [Your answer based on available content]

            **Key Points:**
            - Point 1 (with source info if available)
            - Point 2 (with source info if available)

            **Sources:**
            1. [URL 1] - [Full content / Snippet only]
            2. [URL 2] - [Full content / Snippet only]
            
            **Note:** [If applicable: "Full page content was unavailable; answer based on search snippets"]

            Be accurate, honest about your sources, and always cite URLs.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
        
        return searcher, crawler, synthesizer
    
    def create_tasks(self, query: str, searcher, crawler, synthesizer):
        """Create tasks for the 3-agent workflow"""
        
        # Task 1: Execute web search
        search_task = Task(
            description=f"""Execute a web search for: "{query}"

Your job:
1. Use the web_search tool with this query (or an optimized version)
2. Report back ALL URLs found in the search results
3. Make sure to return the complete formatted output from the tool

Do NOT skip the search. Do NOT answer from your knowledge.
Just search and report the URLs.""",
            agent=searcher,
            expected_output="Search results with at least 3-5 URLs clearly listed"
        )
        
        # Task 2: Crawl top 2 URLs with fallback to snippets
        crawl_task = Task(
            description=f"""Crawl web pages from the search results. Try crawling URLs one by one, and if both fail, use the snippets.

The search results from the previous task show URLs in this format:
[1] Title
    URL: https://example.com/page1
    Snippet: Brief description...
[2] Title  
    URL: https://example.com/page2
    Snippet: Brief description...

YOUR WORKFLOW (FOLLOW STRICTLY):

STEP 1: Try to crawl the FIRST URL
- Find the URL after "[1]" and "URL:" 
- Copy it EXACTLY as shown (no modifications)
- Call: scrape_webpage("exact_url_here")
- Check the result:
  * If it starts with "SUCCESS" → Great! Move to Step 2
  * If it starts with "CRAWL_FAILED" → Note the failure, move to Step 2

STEP 2: Try to crawl the SECOND URL
- Find the URL after "[2]" and "URL:"
- Copy it EXACTLY as shown
- Call: scrape_webpage("exact_url_here")
- Check the result:
  * If it starts with "SUCCESS" → Great! Move to Step 3
  * If it starts with "CRAWL_FAILED" → Note the failure, move to Step 3

STEP 3: Prepare final output
If you successfully crawled AT LEAST ONE URL:
  - Return the crawled content(s) with their SOURCE URLs
  
If BOTH URLs failed to crawl:
  - Return a message: "CRAWL_FAILED_USING_SNIPPETS"
  - Include the snippets from the search results
  - Note which URLs failed and why

CRITICAL RULES:
- Try EACH URL only ONCE - don't retry the same URL
- Copy URLs EXACTLY as they appear (including http:// or https://)
- Don't modify URLs (no adding/removing slashes, params, etc.)
- If a URL fails, acknowledge it and move to the next
- Maximum 2 crawl attempts total (one per URL)

Example:
If scrape_webpage("https://docs.python.org/3/whatsnew/3.13.html") returns:
"CRAWL_FAILED: ..." 

Then immediately try the next URL:
scrape_webpage("https://realpython.com/python313-new-features/")

Don't retry the first URL!""",
            agent=crawler,
            expected_output="Crawled content from 1-2 URLs, OR snippets if both crawls failed, with clear indication of what succeeded/failed",
            context=[search_task]
        )
        
        # Task 3: Synthesize answer from crawled content OR snippets
        synthesize_task = Task(
            description=f"""Answer the question: "{query}"

STEP 1: Check what content is available from the crawler
- Look for "SUCCESS" messages : You have full crawled content
- Look for "CRAWL_FAILED" or "CRAWL_FAILED_USING_SNIPPETS" : Use snippets instead

STEP 2: Create your answer based on available content
If you have crawled content:
  - Use the detailed information from the crawled pages
  - Extract specific facts, features, details
  - This is the preferred source - most reliable

If you only have snippets:
  - Use information from the snippets shown in search results
  - Snippets are the brief text shown under each URL in search results
  - Less detailed but still useful

STEP 3: Format your response

**Answer:**
[Comprehensive answer based on available content - crawled or snippets]

**Key Points:**
- [Specific point from your sources]
- [Another specific point from your sources]
- [etc.]

**Sources:**
1. [First URL] - [Full content crawled / Snippet only]
2. [Second URL] - [Full content crawled / Snippet only]

**Data Quality Note:**
[If using crawled content: "Answer based on full webpage content"]
[If using snippets: "Full page crawling was unavailable - answer based on search result snippets"]

CRITICAL RULES:
- Use ONLY information from the provided content (crawled pages or snippets)
- Do NOT add information from your training data
- Do NOT hallucinate or make up facts
- ALWAYS cite the source URLs
- Be honest about whether you used full content or just snippets
- If the available information is insufficient, say so clearly""",
            agent=synthesizer,
            expected_output="Comprehensive answer with clear source citations and data quality note",
            context=[search_task, crawl_task]
        )
        
        return [search_task, crawl_task, synthesize_task]
    
    def search(self, query: str, verbose: bool = False) -> str:
        """
        Execute web search for a query
        
        Args:
            query: The search query
            verbose: Print detailed progress
            
        Returns:
            Formatted answer with sources
        """
        print(f"\n{'='*70}")
        print(f"🔍 Query: {query}")
        print(f"{'='*70}\n")
        
        # Create agents and tasks
        searcher, crawler, synthesizer = self.create_agents()
        tasks = self.create_tasks(query, searcher, crawler, synthesizer)
        
        # Create and run crew
        crew = Crew(
            agents=[searcher, crawler, synthesizer],
            tasks=tasks,
            process=Process.sequential,
            verbose=verbose,
        )
        
        try:
            result = crew.kickoff()
            
            # Print usage stats
            print(f"\n{'='*70}")
            print("📊 API Usage Statistics:")
            print(f"  Web searches performed: {len(self.search_cache)}")
            print(f"  Pages crawled: {len(self.crawl_cache)}")
            if self.crawl_cache:
                print(f"  Crawled URLs:")
                for url in self.crawl_cache.keys():
                    print(f"    - {url}")
            print(f"{'='*70}\n")
            
            return str(result)
            
        except Exception as e:
            return f"Error during search: {str(e)}"
    
    def clear_cache(self):
        """Clear search and crawl caches"""
        self.search_cache.clear()
        self.crawl_cache.clear()
        print("✅ Cache cleared")
