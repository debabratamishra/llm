"""
Google ADK (Agent Development Kit) with Ollama Backend
Demonstrates different agent workflows using gemma3n:e2b model from Ollama

Note: Uses OpenAI-compatible API since Google ADK doesn't natively support Ollama
"""

import os
from typing import List, Dict, Any, Optional
import json
from openai import OpenAI


class OllamaADKAgent:
    """Base class for ADK agents using Ollama backend via OpenAI-compatible API"""
    
    def __init__(self, model_name: str = "gemma3n:e2b", base_url: str = "http://localhost:11434/v1"):
        """
        Initialize the ADK agent with Ollama backend
        
        Args:
            model_name: Name of the Ollama model (default: gemma3n:e2b)
            base_url: Ollama API endpoint (OpenAI-compatible)
        """
        self.model_name = model_name
        self.base_url = base_url
        
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama"
        )
        
    def chat(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Basic chat interaction
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            
        Returns:
            Model response text
        """
        messages = []
        
        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        )
        
        return response.choices[0].message.content or ""


class SimpleQAAgent(OllamaADKAgent):
    """Simple Question-Answering Agent"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """You are a helpful AI assistant. 
        Provide clear, concise, and accurate answers to user questions.
        If you don't know something, admit it rather than making up information."""
        
    def ask(self, question: str) -> str:
        """Ask a question and get an answer"""
        return self.chat(question, self.system_prompt)


class ConversationalAgent(OllamaADKAgent):
    """Multi-turn conversational agent with memory"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = """You are a friendly conversational AI assistant.
        Maintain context across the conversation and provide helpful responses."""
        
    def chat_with_history(self, user_message: str) -> str:
        """
        Chat with conversation history
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response
        """
        # Build conversation context
        context = self._build_context()
        full_prompt = f"{context}\n\nUser: {user_message}\nAssistant:"
        
        # Get response
        response = self.chat(full_prompt, self.system_prompt)
        
        # Update history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _build_context(self) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return ""
        
        context_lines = []
        for msg in self.conversation_history[-6:]:  # Keep last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_lines)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class TaskPlanningAgent(OllamaADKAgent):
    """Agent that breaks down complex tasks into steps"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """You are a task planning expert.
        When given a complex task, break it down into clear, actionable steps.
        Number each step and provide specific details."""
        
    def plan_task(self, task_description: str) -> Dict[str, Any]:
        """
        Create a plan for a complex task
        
        Args:
            task_description: Description of the task
            
        Returns:
            Dictionary with task plan
        """
        prompt = f"""Please break down the following task into detailed steps:

Task: {task_description}

Provide a numbered list of steps with specific actions for each."""
        
        response = self.chat(prompt, self.system_prompt)
        
        return {
            "task": task_description,
            "plan": response,
            "steps": self._parse_steps(response)
        }
    
    def _parse_steps(self, plan_text: str) -> List[str]:
        """Parse numbered steps from plan text"""
        steps = []
        for line in plan_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                steps.append(line)
        return steps


class RAGAgent(OllamaADKAgent):
    """Retrieval-Augmented Generation Agent"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base: List[Dict[str, str]] = []
        self.system_prompt = """You are a RAG (Retrieval-Augmented Generation) assistant.
        Answer questions based on the provided context. If the context doesn't contain
        the answer, say so clearly."""
        
    def add_document(self, title: str, content: str):
        """Add a document to the knowledge base"""
        self.knowledge_base.append({
            "title": title,
            "content": content
        })
    
    def query(self, question: str, top_k: int = 3) -> str:
        """
        Query with retrieval-augmented generation
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            Generated answer based on retrieved context
        """
        # Simple keyword-based retrieval (in production, use embeddings)
        retrieved_docs = self._retrieve_documents(question, top_k)
        
        # Build context
        context = "\n\n".join([
            f"Document: {doc['title']}\n{doc['content']}"
            for doc in retrieved_docs
        ])
        
        # Generate answer
        prompt = f"""Context:
{context}

Question: {question}

Please answer the question based on the context provided above."""
        
        response = self.chat(prompt, self.system_prompt)
        
        return response
    
    def _retrieve_documents(self, query: str, top_k: int) -> List[Dict[str, str]]:
        """Simple keyword-based document retrieval"""
        query_terms = set(query.lower().split())
        
        # Score documents by keyword overlap
        scored_docs = []
        for doc in self.knowledge_base:
            doc_terms = set(doc['content'].lower().split())
            score = len(query_terms & doc_terms)
            scored_docs.append((score, doc))
        
        # Sort by score and return top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:top_k]]


class CodeAssistantAgent(OllamaADKAgent):
    """Agent specialized for code-related tasks"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """You are an expert programming assistant.
        Help users with code writing, debugging, and explanation.
        Provide clear, well-commented code examples."""
        
    def generate_code(self, description: str, language: str = "python") -> str:
        """Generate code from description"""
        prompt = f"""Generate {language} code for the following:

{description}

Provide clean, well-commented code."""
        
        return self.chat(prompt, self.system_prompt)
    
    def explain_code(self, code: str) -> str:
        """Explain what code does"""
        prompt = f"""Explain what this code does:

```
{code}
```

Provide a clear explanation of its functionality."""
        
        return self.chat(prompt, self.system_prompt)
    
    def debug_code(self, code: str, error: Optional[str] = None) -> str:
        """Help debug code"""
        prompt = f"""Debug this code:

```
{code}
```
"""
        if error:
            prompt += f"\nError message: {error}"
            
        prompt += "\n\nIdentify the issue and suggest fixes."
        
        return self.chat(prompt, self.system_prompt)


class ChainOfThoughtAgent(OllamaADKAgent):
    """Agent that uses chain-of-thought reasoning"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """You are a reasoning expert that thinks step-by-step.
        For complex questions, break down your thinking process clearly.
        Show your work before arriving at the final answer."""
        
    def reason(self, problem: str) -> Dict[str, Any]:
        """
        Solve problem using chain-of-thought reasoning
        
        Args:
            problem: Problem to solve
            
        Returns:
            Dictionary with reasoning steps and answer
        """
        prompt = f"""Let's solve this step by step:

{problem}

Think through this carefully, showing your reasoning at each step."""
        
        response = self.chat(prompt, self.system_prompt)
        
        return {
            "problem": problem,
            "reasoning": response,
            "steps": self._extract_steps(response)
        }
    
    def _extract_steps(self, reasoning: str) -> List[str]:
        """Extract reasoning steps from response"""
        steps = []
        for line in reasoning.split('\n'):
            line = line.strip()
            if line and ('step' in line.lower() or line[0].isdigit()):
                steps.append(line)
        return steps


# Demonstration Functions
def demo_simple_qa():
    """Demonstrate simple Q&A agent"""
    print("\n" + "="*60)
    print("DEMO 1: Simple Q&A Agent")
    print("="*60)
    
    agent = SimpleQAAgent()
    
    questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What are the benefits of exercise?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        answer = agent.ask(q)
        print(f"A: {answer}")


def demo_conversational():
    """Demonstrate conversational agent with memory"""
    print("\n" + "="*60)
    print("DEMO 2: Conversational Agent with Memory")
    print("="*60)
    
    agent = ConversationalAgent()
    
    conversation = [
        "Hi! My name is Alice.",
        "What's my name?",
        "What are some good hobbies to learn?",
        "Which of those would be good for relaxation?"
    ]
    
    for message in conversation:
        print(f"\nUser: {message}")
        response = agent.chat_with_history(message)
        print(f"Assistant: {response}")


def demo_task_planning():
    """Demonstrate task planning agent"""
    print("\n" + "="*60)
    print("DEMO 3: Task Planning Agent")
    print("="*60)
    
    agent = TaskPlanningAgent()
    
    task = "Build a web application for a todo list"
    print(f"\nTask: {task}")
    
    plan = agent.plan_task(task)
    print(f"\nPlan:\n{plan['plan']}")
    print(f"\nExtracted Steps ({len(plan['steps'])} total):")
    for step in plan['steps'][:5]:  # Show first 5
        print(f"  {step}")


def demo_rag():
    """Demonstrate RAG agent"""
    print("\n" + "="*60)
    print("DEMO 4: Retrieval-Augmented Generation Agent")
    print("="*60)
    
    agent = RAGAgent()
    
    # Add documents to knowledge base
    agent.add_document(
        "Python Basics",
        "Python is a high-level programming language. It uses indentation for code blocks. "
        "Variables don't need type declarations. Python supports multiple programming paradigms."
    )
    agent.add_document(
        "Python Data Structures",
        "Python has built-in data structures like lists, tuples, dictionaries, and sets. "
        "Lists are mutable ordered collections. Dictionaries store key-value pairs."
    )
    agent.add_document(
        "JavaScript Overview",
        "JavaScript is a programming language primarily used for web development. "
        "It runs in browsers and on servers via Node.js. JavaScript uses curly braces for blocks."
    )
    
    question = "What are Python's built-in data structures?"
    print(f"\nQuestion: {question}")
    
    answer = agent.query(question)
    print(f"\nAnswer: {answer}")


def demo_code_assistant():
    """Demonstrate code assistant agent"""
    print("\n" + "="*60)
    print("DEMO 5: Code Assistant Agent")
    print("="*60)
    
    agent = CodeAssistantAgent()
    
    # Generate code
    print("\n--- Code Generation ---")
    description = "A function that calculates the fibonacci sequence up to n terms"
    print(f"Request: {description}")
    code = agent.generate_code(description)
    print(f"\nGenerated Code:\n{code}")
    
    # Explain code
    print("\n--- Code Explanation ---")
    sample_code = "lambda x: x**2 if x > 0 else -x**2"
    print(f"Code: {sample_code}")
    explanation = agent.explain_code(sample_code)
    print(f"\nExplanation: {explanation}")


def demo_chain_of_thought():
    """Demonstrate chain-of-thought reasoning"""
    print("\n" + "="*60)
    print("DEMO 6: Chain-of-Thought Reasoning Agent")
    print("="*60)
    
    agent = ChainOfThoughtAgent()
    
    problem = "If a train travels 120 km in 2 hours, how far will it travel in 5 hours at the same speed?"
    print(f"\nProblem: {problem}")
    
    result = agent.reason(problem)
    print(f"\nReasoning:\n{result['reasoning']}")


def main():
    """Run all demonstrations"""
    print("\n" + "🤖 "*30)
    print("Google ADK Agent Workflows with Ollama Backend")
    print("Model: gemma3n:e2b")
    print("🤖 "*30)
    
    # Run demos
    try:
        demo_simple_qa()
        demo_conversational()
        demo_task_planning()
        demo_rag()
        demo_code_assistant()
        demo_chain_of_thought()
        
        print("\n" + "="*60)
        print("All demonstrations completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\nMake sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. Model 'gemma3n:e2b' is available (ollama list)")
        print("3. Google ADK is properly installed")


if __name__ == "__main__":
    main()
