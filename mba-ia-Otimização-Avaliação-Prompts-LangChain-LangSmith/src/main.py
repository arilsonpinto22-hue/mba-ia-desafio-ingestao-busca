import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

def get_llm(model_type="runner"):
    """
    Factory function to get the appropriate LLM based on provider and type.
    model_type: "optimizer", "runner", or "evaluator"
    """
    provider = os.getenv("PROVIDER", "openai").lower()
    
    if provider == "gemini":
        # Google Gemini Configuration
        if model_type == "runner":
            model_name = os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash")
        elif model_type == "evaluator":
            model_name = os.getenv("GEMINI_LLM_EVALUATOR", "gemini-1.5-flash")
        else: # optimizer - uses evaluator model for intelligence
            model_name = os.getenv("GEMINI_LLM_EVALUATOR", "gemini-1.5-flash")
            
        print(f"Initializing Gemini [{model_type}]: {model_name}")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
        
    else:
        # OpenAI Configuration (Default)
        if model_type == "runner":
            model_name = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        elif model_type == "evaluator":
            model_name = os.getenv("OPENAI_LLM_EVALUATOR", "gpt-4o")
        else: # optimizer
            model_name = os.getenv("OPENAI_LLM_EVALUATOR", "gpt-4o")
            
        print(f"Initializing OpenAI [{model_type}]: {model_name}")
        return ChatOpenAI(model=model_name, temperature=0)

# Initialize LLMs dynamically
llm_optimizer = get_llm("optimizer")
llm_runner = get_llm("runner")
llm_evaluator = get_llm("evaluator")

def create_bad_prompt():
    """Returns the repo name for the target prompt."""
    return "leonanluppi/bug_to_user_story_v1"

def pull_prompt(repo_name):
    """Pulls a prompt from LangSmith Hub."""
    print(f"Pulling prompt from {repo_name}...")
    try:
        # Try to pull using hub
        from langchain import hub
        prompt = hub.pull(repo_name)
        return prompt
    except Exception as e:
        print(f"Error pulling prompt: {e}")
        # Fallback to the raw prompt content we know
        print("Using local fallback for bug_to_user_story_v1")
        return ChatPromptTemplate.from_template(
            """Você é um assistente que ajuda a transformar relatos de bugs de usuários em tarefas para desenvolvedores.

            Analise o relato de bug abaixo e crie uma user story a partir dele.

            Relato de Bug:
            ---
            {bug_report}
            ---

            User Story gerada:
            """
        )

def optimize_prompt(original_prompt_text):
    """Optimizes the prompt using advanced techniques (Role, Few-Shot, CoT)."""
    print("Optimizing prompt using advanced techniques (Role, Few-Shot, CoT)...")
    
    # Meta-Prompt that instructs the LLM to build a sophisticated prompt
    optimization_prompt = ChatPromptTemplate.from_template(
        """You are a Senior Prompt Engineer. 
        Your task is to refactor the following "Bug to User Story" prompt to make it world-class.
        
        Apply the following techniques:
        1. **Role Prompting**: Define a persona (e.g., Senior Product Owner).
        2. **Few-Shot Learning**: Include at least one clear example of Input (Bug) -> Output (User Story + Acceptance Criteria).
        3. **Chain of Thought**: Instruct the model to step-by-step analyze the bug before writing the story.
        
        Original Prompt:
        "{original_prompt}"
        
        The new prompt must accept the input variable: {{bug_report}}.
        
        Output ONLY the optimized prompt template.
        """
    )
    
    chain = optimization_prompt | llm_optimizer
    optimized_text = chain.invoke({"original_prompt": original_prompt_text}).content
    
    # Clean up
    optimized_text = optimized_text.replace("```markdown", "").replace("```", "").strip()
    
    print(f"Optimized Prompt: \n{optimized_text}\n")
    return ChatPromptTemplate.from_template(optimized_text)

def evaluate_metrics(prompt_template, sample_inputs):
    """
    Evaluates the prompt using LLM-as-a-Judge for F1-Score, Clarity, and Precision.
    """
    print("Evaluating prompt metrics...")
    
    prompt_text = prompt_template.messages[0].prompt.template
    
    # We will average the scores over the sample inputs
    total_scores = {"f1": 0.0, "clarity": 0.0, "precision": 0.0}
    
    for inp in sample_inputs:
        # 1. Generate Output (Using the target runner model)
        try:
            chain = prompt_template | llm_runner
            output = chain.invoke(inp).content
        except Exception as e:
            print(f"Execution failed for input {inp}: {e}")
            continue

        # 2. Evaluate Output & Prompt Pair (Using the evaluator model)
        eval_prompt = ChatPromptTemplate.from_template(
            """You are a Senior QA Specialist. Evaluate the quality of the User Story generated from a Bug Report.
            
            Prompt Used: "{prompt_text}"
            Bug Report Input: {input}
            Generated User Story: "{output}"
            
            Rate the following metrics on a scale from 0.0 to 1.0:
            
            1. Clarity (0.0-1.0): Is the User Story clear, following standard format (As a, I want, So that)?
            2. Precision (0.0-1.0): Does it accurately reflect the bug reported without adding unrelated features?
            3. F1-Score (0.0-1.0): (Proxy) Does it include Acceptance Criteria that cover the fix verification?
            
            Return the scores in JSON format: {{"clarity": float, "precision": float, "f1": float}}
            """
        )
        
        eval_chain = eval_prompt | llm_evaluator
        result_str = eval_chain.invoke({
            "prompt_text": prompt_text,
            "input": inp,
            "output": output
        }).content
        
        try:
            start = result_str.find('{')
            end = result_str.rfind('}') + 1
            scores = json.loads(result_str[start:end])
            
            total_scores["f1"] += scores.get("f1", 0)
            total_scores["clarity"] += scores.get("clarity", 0)
            total_scores["precision"] += scores.get("precision", 0)
            
        except Exception as e:
            print(f"Error parsing scores: {e}")
    
    # Average
    num_samples = len(sample_inputs)
    avg_scores = {k: round(v / num_samples, 2) for k, v in total_scores.items()}
    
    return avg_scores

def main():
    # 1. Setup & Pull
    repo_name = create_bad_prompt()
    if not repo_name:
        repo_name = "bad-prompt-example-v1" # Fallback
        
    bad_prompt = pull_prompt(repo_name)
    
    # Extract text (handling different prompt types simply)
    try:
        # Check if it's a ChatPromptTemplate with messages
        if hasattr(bad_prompt, 'messages') and bad_prompt.messages:
            original_text = bad_prompt.messages[0].prompt.template
        # Check if it's a StringPromptTemplate (e.g. PromptTemplate)
        elif hasattr(bad_prompt, 'template'):
            original_text = bad_prompt.template
        else:
            original_text = str(bad_prompt)
    except:
        original_text = "Analyze bug: {bug_report}"

    print(f"Original Prompt: {original_text}")

    # 2. Optimize
    # We loop until we hit the target or max retries
    target_score = 0.9
    max_retries = 3
    current_prompt = bad_prompt
    current_text = original_text
    
    # Define sample inputs for evaluation
    sample_inputs = [
        {"bug_report": "When I click the 'Buy' button on the checkout page, nothing happens and the spinner keeps loading forever. I am on Chrome v90."},
        {"bug_report": "The profile image uploads but shows as broken link on the dashboard. It works fine on the settings page."}
    ]

    for i in range(max_retries):
        print(f"\n--- Optimization Round {i+1} ---")
        
        if i > 0:
            # Re-optimize based on previous feedback (simplified here to just re-running optimizer)
            optimized_prompt = optimize_prompt(current_text)
        else:
            optimized_prompt = optimize_prompt(current_text)
            
        # 3. Evaluate
        scores = evaluate_metrics(optimized_prompt, sample_inputs)
        print(f"Scores: {scores}")
        
        if all(s >= target_score for s in scores.values()):
            print("\nSUCCESS: Target scores achieved!")
            
            # 4. Push
            new_repo_name = f"{repo_name}-optimized"
            try:
                from langchain import hub
                url = hub.push(new_repo_name, optimized_prompt)
                print(f"Pushed optimized prompt to {url}")
            except Exception as e:
                print(f"Error pushing optimized prompt: {e}")
            break
        else:
            print("Target not met. Retrying...")
            # Set current text to the optimized one to refine further? 
            # Or keep the original and ask for better?
            # Usually better to iterate on the optimized one or give feedback.
            # For simplicity, we just try to optimize the *result* of the last round.
            current_prompt = optimized_prompt
            current_text = current_prompt.messages[0].prompt.template
            
    else:
        print("\nFAILED: Max retries reached without hitting target scores.")

if __name__ == "__main__":
    main()