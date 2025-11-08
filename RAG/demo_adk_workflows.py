"""
Google ADK Agent Workflows with Ollama Backend
===============================================

QUICK START:
    python demo_adk_workflows.py          # Interactive menu
    python demo_adk_workflows.py 1        # Run specific demo (1-7)
    
REQUIREMENTS:
    - Ollama running: http://localhost:11434
    - Model: gemma3n:e2b

AGENTS AVAILABLE:
    1. SimpleQAAgent        - Direct question answering
    2. ConversationalAgent  - Multi-turn dialogue with memory
    3. TaskPlanningAgent    - Complex task breakdown into steps
    4. RAGAgent            - Knowledge-base Q&A
    5. CodeAssistantAgent  - Code generation/explanation/debugging
    6. ChainOfThoughtAgent - Step-by-step reasoning

PROGRAMMATIC USAGE:
    from adk_ollama import SimpleQAAgent
    
    agent = SimpleQAAgent()
    answer = agent.ask("What is Python?")
    print(answer)

For full implementation details, see adk_ollama.py
"""

import sys
sys.path.insert(0, '/Users/debabratamishra/Code/llm/RAG')

from adk_ollama import (
    SimpleQAAgent,
    ConversationalAgent,
    TaskPlanningAgent,
    RAGAgent,
    CodeAssistantAgent,
    ChainOfThoughtAgent
)


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def demo_1_simple_qa():
    """Demo 1: Simple Question-Answering Agent"""
    print_header("DEMO 1: Simple Q&A Agent")
    print("\nThis agent answers straightforward questions.\n")
    
    agent = SimpleQAAgent()
    
    questions = [
        "What is Python?",
        "Name three benefits of meditation.",
        "Who invented the telephone?"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n[Question {i}] {q}")
        answer = agent.ask(q)
        print(f"[Answer] {answer}\n")
        print("-" * 70)


def demo_2_conversational():
    """Demo 2: Conversational Agent with Memory"""
    print_header("DEMO 2: Conversational Agent with Memory")
    print("\nThis agent maintains context across multiple exchanges.\n")
    
    agent = ConversationalAgent()
    
    conversation = [
        "Hi! I'm planning a trip to Japan next month.",
        "What are some must-visit places?",
        "Which of those is best for food lovers?",
        "Thanks! Can you remind me where I said I'm going?"
    ]
    
    for i, message in enumerate(conversation, 1):
        print(f"\n[Turn {i}] User: {message}")
        response = agent.chat_with_history(message)
        print(f"[Turn {i}] Assistant: {response}")
        print("-" * 70)


def demo_3_task_planning():
    """Demo 3: Task Planning Agent"""
    print_header("DEMO 3: Task Planning Agent")
    print("\nThis agent breaks down complex tasks into actionable steps.\n")
    
    agent = TaskPlanningAgent()
    
    tasks = [
        "Learn to play guitar",
        "Start a podcast"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[Task {i}] {task}")
        plan = agent.plan_task(task)
        print(f"\n{plan['plan']}")
        print(f"\n→ Extracted {len(plan['steps'])} steps")
        print("-" * 70)


def demo_4_rag():
    """Demo 4: Retrieval-Augmented Generation (RAG) Agent"""
    print_header("DEMO 4: RAG Agent (Retrieval-Augmented Generation)")
    print("\nThis agent answers questions based on a knowledge base.\n")
    
    agent = RAGAgent()
    
    # Build knowledge base
    print("[Building Knowledge Base...]")
    agent.add_document(
        "Machine Learning Basics",
        "Machine learning is a subset of AI that enables computers to learn from data. "
        "It includes supervised learning (with labeled data), unsupervised learning "
        "(finding patterns), and reinforcement learning (learning through trial and error). "
        "Common algorithms include neural networks, decision trees, and support vector machines."
    )
    agent.add_document(
        "Deep Learning",
        "Deep learning uses neural networks with multiple layers to process complex patterns. "
        "It powers applications like image recognition, natural language processing, and speech recognition. "
        "Popular frameworks include TensorFlow, PyTorch, and Keras. CNNs are used for images, "
        "RNNs and Transformers for sequences."
    )
    agent.add_document(
        "Natural Language Processing",
        "NLP enables computers to understand and generate human language. "
        "It includes tasks like text classification, sentiment analysis, machine translation, "
        "and question answering. Modern NLP uses transformer models like BERT, GPT, and T5. "
        "Applications include chatbots, search engines, and content generation."
    )
    print("✓ Knowledge base ready with 3 documents\n")
    
    # Ask questions
    questions = [
        "What is machine learning?",
        "What frameworks are used for deep learning?",
        "What are some NLP applications?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Query {i}] {question}")
        answer = agent.query(question, top_k=2)
        print(f"[Answer] {answer}")
        print("-" * 70)


def demo_5_code_assistant():
    """Demo 5: Code Assistant Agent"""
    print_header("DEMO 5: Code Assistant Agent")
    print("\nThis agent helps with code generation, explanation, and debugging.\n")
    
    agent = CodeAssistantAgent()
    
    # Code generation
    print("\n[Task A] Generate Code")
    description = "A function that checks if a string is a palindrome"
    print(f"Request: {description}\n")
    code = agent.generate_code(description)
    print(f"Generated Code:\n{code}")
    print("-" * 70)
    
    # Code explanation
    print("\n[Task B] Explain Code")
    sample_code = "list(map(lambda x: x**2, range(10)))"
    print(f"Code: {sample_code}\n")
    explanation = agent.explain_code(sample_code)
    print(f"Explanation:\n{explanation}")
    print("-" * 70)
    
    # Code debugging
    print("\n[Task C] Debug Code")
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
    
result = calculate_average([])
"""
    print(f"Code:\n{buggy_code}\n")
    debug_help = agent.debug_code(buggy_code, "ZeroDivisionError: division by zero")
    print(f"Debugging Help:\n{debug_help}")
    print("-" * 70)


def demo_6_chain_of_thought():
    """Demo 6: Chain-of-Thought Reasoning Agent"""
    print_header("DEMO 6: Chain-of-Thought Reasoning Agent")
    print("\nThis agent shows its reasoning process step-by-step.\n")
    
    agent = ChainOfThoughtAgent()
    
    problems = [
        "If a book costs $12 and you get a 25% discount, how much do you pay?",
        "There are 3 boxes. Each box contains 4 smaller boxes, and each smaller box contains 5 toys. How many toys in total?"
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n[Problem {i}] {problem}\n")
        result = agent.reason(problem)
        print(f"Reasoning:\n{result['reasoning']}")
        print(f"\n→ Identified {len(result['steps'])} reasoning steps")
        print("-" * 70)


def show_menu():
    """Display demo menu"""
    print("\n" + "🤖" * 35)
    print("Google ADK Agent Workflows with Ollama Backend")
    print("Model: gemma3n:e2b")
    print("🤖" * 35)
    print("\nAvailable Demonstrations:")
    print("  1. Simple Q&A Agent")
    print("  2. Conversational Agent with Memory")
    print("  3. Task Planning Agent")
    print("  4. RAG (Retrieval-Augmented Generation) Agent")
    print("  5. Code Assistant Agent")
    print("  6. Chain-of-Thought Reasoning Agent")
    print("  7. Run All Demos")
    print("  0. Exit")
    print("\n" + "="*70)


def run_demo(choice):
    """Run the selected demo"""
    demos = {
        '1': demo_1_simple_qa,
        '2': demo_2_conversational,
        '3': demo_3_task_planning,
        '4': demo_4_rag,
        '5': demo_5_code_assistant,
        '6': demo_6_chain_of_thought,
    }
    
    if choice == '7':
        for demo_func in demos.values():
            try:
                demo_func()
            except Exception as e:
                print(f"\n❌ Error in demo: {e}")
                import traceback
                traceback.print_exc()
    elif choice in demos:
        try:
            demos[choice]()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ Invalid choice!")


def main():
    """Main interactive menu"""
    while True:
        show_menu()
        choice = input("Enter your choice (0-7): ").strip()
        
        if choice == '0':
            print("\n👋 Thank you for exploring ADK agents with Ollama!")
            print("="*70)
            break
        
        run_demo(choice)
        
        print("\n" + "="*70)
        input("Press Enter to continue...")


if __name__ == "__main__":
    # Check if running with command-line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        run_demo(choice)
    else:
        main()
