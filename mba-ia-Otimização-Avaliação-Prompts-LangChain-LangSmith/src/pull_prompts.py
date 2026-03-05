import os
import yaml
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from the root .env
load_dotenv()

def pull_and_save_prompts():
    """
    Pulls specific prompts from LangSmith and saves them locally.
    """
    # Define target prompts
    target_prompts = [
        "leonanluppi/bug_to_user_story_v1"
    ]
    
    # Ensure output directory exists
    output_dir = "prompts"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "raw_prompts.yml")
    
    collected_data = []

    print(f"--- Starting Pull Process ---")
    
    client = Client()
    
    for repo_name in target_prompts:
        print(f"Pulling: {repo_name}...")
        try:
            # Pull from LangSmith Hub
            prompt_obj = client.pull_prompt(repo_name)
            
            # Extract template content safely
            content = ""
            if hasattr(prompt_obj, 'messages'):
                # Handle ChatPromptTemplate
                for msg in prompt_obj.messages:
                    if hasattr(msg, 'prompt'):
                        content += msg.prompt.template + "\n"
                    else:
                        content += str(msg) + "\n"
            elif hasattr(prompt_obj, 'template'):
                # Handle StringPromptTemplate
                content = prompt_obj.template
            else:
                content = str(prompt_obj)

            # Structure data
            prompt_data = {
                "name": repo_name,
                "input_variables": prompt_obj.input_variables,
                "template": content.strip()
            }
            collected_data.append(prompt_data)
            print(f"✅ Successfully pulled {repo_name}")
            
        except Exception as e:
            print(f"❌ Error pulling {repo_name}: {e}")
            # Create a dummy entry if pull fails (e.g., auth error) for testing flow
            print("  -> Creating fallback entry for continuity...")
            collected_data.append({
                "name": repo_name,
                "input_variables": ["bug_report"],
                "template": "Analyze this bug report and convert it to a user story: {bug_report} (FALLBACK)",
                "note": "Pull failed, using fallback."
            })

    # Save to YAML
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(collected_data, f, allow_unicode=True, sort_keys=False)
        print(f"\nSaved {len(collected_data)} prompts to: {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    pull_and_save_prompts()