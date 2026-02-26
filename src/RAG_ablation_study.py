"""
RAG Ablation Study for SOMA VLM Perception

This script conducts a comprehensive ablation study to evaluate the impact of Retrieval-Augmented Generation (RAG) on the perception module of the SOMA agent.
It simulates three levels of RAG experience (No RAG, Limited RAG, Rich RAG) and evaluates the generated perception plans using a VLM-based scoring system across multiple iterations for robustness.
The final results, including average scores and sample plans, are saved to a JSON file for analysis.
"""
import json
import logging
import statistics
import re
from soma_vlm import Qwen3VLAPIClient #

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import json
import logging
import re
from soma_vlm import Qwen3VLAPIClient

class PerceptionEvaluator:
    def __init__(self, vlm_client: Qwen3VLAPIClient):
        self.vlm = vlm_client

    def _get_system_prompt_for_task(self, task_type: str) -> str:
        """VLM Expert Score: Get score for SOMA VLM Output"""
        
        base_guideline = (
            "You are a strict Robotics Evaluator grading a VLM's perception plan out of 100 points.\n"
            "You MUST score the plan across these 3 dimensions:\n\n"
            "1. Task_Execution_and_Tools (0-40 points): Did it choose the logically correct tool and specify accurate parameters for the original task?\n"
            "2. Experience_Adherence (0-40 points) -> [CRITICAL ABLATION RULE]:\n"
            "   - If 'Historical Experience Provided' is exactly 'NONE': You MUST score exactly 0 points here. No exceptions.\n"
            "   - If 'Historical Experience Provided' is NOT 'NONE': Score how perfectly the plan adopted the specific advice (e.g., using specific target names, specific tool chains, or exact subtask phrasing). Give 35-40 for perfect adherence, 15-25 for partial adherence.\n"
            "3. Efficiency_and_Precision (0-20 points): Is the 'refined_task' clean? Did it avoid redundant tools or hallucinated parameters? (Deduct points for rambling verbosity).\n\n"
        )
        
        prompts = {
            "1_visual_overlay": base_guideline + (
                "Specific Focus: Target is the center bowl. Experience dictates using 'replace_texture' to render it white.\n"
            ),
            "2_distractor_remove": base_guideline + (
                "Specific Focus: Must pick leftmost bowl and explicitly list the other 4 bowls individually in 'objects_to_remove'.\n"
            ),
            "3_noisy_prompt": base_guideline + (
                "Specific Focus: Denoise rambling verbal instructions. Use 'encore' to avoid unnecessary visual mods. Refined task must be strict 'Pick X and place in Y'.\n"
            ),
            "4_long_task_subtask": base_guideline + (
                "Specific Focus: Task decomposition. Must use 'task_decompose' and list EXACTLY 3 atomic subtasks with correct item-receptacle pairings.\n"
            )
        }
        
        json_format = (
            "\nYou MUST output ONLY a valid JSON object in this exact format:\n"
            "{\n"
            "  \"Task_Execution_Score\": 35,\n"
            "  \"Experience_Adherence_Score\": 40,\n"
            "  \"Efficiency_Score\": 18,\n"
            "  \"Total\": 93,\n"
            "  \"Reasoning\": \"Brief explanation...\"\n"
            "}"
        )
        return prompts.get(task_type, prompts["1_visual_overlay"]) + json_format

    def score_plan(self, task_type: str, image_path: str, task_desc: str, plan: dict, rag_context: str) -> dict:
        system_prompt = self._get_system_prompt_for_task(task_type)
        user_prompt = (
            f"Original Task: {task_desc}\n"
            f"Historical Experience Provided: {rag_context if rag_context else 'NONE'}\n"
            f"Generated Plan to Evaluate: {json.dumps(plan, ensure_ascii=False, indent=2)}\n"
        )
        
        b64_img = self.vlm._encode_image(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": b64_img}}
            ]}
        ]
        
        for attempt in range(3):
            # Low Temperature to reduce hallucination and improve JSON compliance
            response_text = self.vlm._generate(messages, max_tokens=300, temperature=0.1)
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    extracted_json = json_match.group(0)
                    result = json.loads(extracted_json)
                    
                    final_result = {
                        "Task_Execution": result.get("Task_Execution_Score", 0),
                        "Experience_Adherence": result.get("Experience_Adherence_Score", 0),
                        "Efficiency": result.get("Efficiency_Score", 0),
                        "Reasoning": result.get("Reasoning", "")
                    }
                    final_result["Total"] = (
                        final_result["Task_Execution"] + 
                        final_result["Experience_Adherence"] + 
                        final_result["Efficiency"]
                    )
                    return final_result
            except Exception as e:
                logging.warning(f"Parse attempt {attempt+1} failed. Retrying...")
                continue
                
        return {"Total": -1, "Reasoning": "Parse error after 3 attempts"}
    
def run_comprehensive_ablation(iterations=3):
    vlm = Qwen3VLAPIClient() #
    evaluator = PerceptionEvaluator(vlm)
    
    test_cases = {
        "1_visual_overlay": {
            "image": "/mnt/disk1/shared_data/lzy/SOMA/src/experiment_step/visual/data/original.jpg",  
            "task": "Pick up the red bowl in the center of the cross and place it on the plate.",   
            
            # --- Simulate Experience Bank 1：No RAG ---
            "mock_no_rag": {
                "context": "",
                "hints": {}
            },

            # --- Simulate Experience Bank 2：Limited RAG ---
            # Only one failure experience, without success experience
            "mock_limited_rag": {
                "context": "Warning! Past failures causes: Visual Interference. The robot was confused by the multiple identical red bowls and grabbed the wrong one.",
                "hints": {
                    "success_has_object_texture": False,
                    "failure_has_object_texture": False
                }
            },
            
            # --- Simulate Experience Bank 3：Rich RAG ---
            # Have both failure and success experience, with detailed diagnosis and clear actionable advice
            "mock_rich_rag": {
                "context": (
                    "Warning! Past failures causes: \n"
                    "1. Distractor Interference: Grabbed the wrong red bowl because all 5 look identical.\n"
                    "2. Visual Ambiguity: Robot twitched over empty space.\n"
                    "[SUCCESS MEMORY] Past success confirms that the policy is highly confident in grasping "
                    "a 'white bowl'. Using 'replace_texture' to render the center red bowl into a white bowl "
                    "results in a 95% success rate."
                ),
                "hints": {
                    "success_has_object_texture": True,
                    "failure_has_object_texture": False
                }
            }
        },
        
        "2_distractor_remove": {
            "image": "/mnt/disk1/shared_data/lzy/SOMA/src/experiment_step/remove-distractor/data/original.jpg",  
            "task": "Pick up the leftmost bowl and place it on the plate.",   
            
            "mock_no_rag": {
                "context": "",
                "hints": {}
            },

            "mock_limited_rag": {
                "context": (
                    "Warning! Past failures causes: \n"
                    "1. Distractor Interference: The robot grabbed the wrong bowl because they all look identical.\n"
                    "2. Perception Error: The vision system accidentally removed the target plate or the target leftmost bowl."
                ),
                "hints": {
                    "success_has_object_texture": False,
                    "failure_has_object_texture": False
                }
            },
            
            "mock_rich_rag": {
                "context": (
                    "Warning! Past failures causes: \n"
                    "1. Distractor Interference: Grabbed the wrong bowl due to identical distractors.\n"
                    "2. Tool Misuse: The vision system accidentally removed the plate, making placement impossible.\n"
                    "3. Incomplete Removal: Using generic phrases like 'the other bowls' failed to remove all 4 distractor bowls, leaving some behind to confuse the robot.\n"
                    "[SUCCESS MEMORY] To achieve a 100% success rate, you MUST use the 'remove_distractor' tool. "
                    "Crucially, you must explicitly and individually list the 4 wrong bowls in the 'objects_to_remove' array "
                    "(e.g., 'the bowl on the top', 'the bowl in the center', 'the bowl on the right', 'the bowl on the bottom'). "
                    "Do NOT remove the plate. Do NOT remove the leftmost bowl."
                ),
                "hints": {
                    "success_has_object_texture": False, 
                    "failure_has_object_texture": False
                }
            }
        },
        
        "3_noisy_prompt": {
            "image": "/mnt/disk1/shared_data/lzy/SOMA/src/experiment_step/noisy-prompt/data/original.jpg",  
            "task": "Hey, umm... look down there. Can you grab that bottle? You know, the one for fries? Yeah, put it in the basket.",   
            
            "mock_no_rag": {
                "context": "",
                "hints": {}
            },

            "mock_limited_rag": {
                "context": (
                    "Warning! Past failures causes: \n"
                    "1. Instruction Misinterpretation: The robot grabbed the wrong object (the cylindrical tomato soup can) instead of the bottle.\n"
                    "2. Non-standard Formatting: The refined prompt remained too verbose or lacked a clear focus, failing to match the policy's expected 'Pick [object] and place it in [receptacle]' format."
                ),
                "hints": {
                    "success_has_object_texture": False,
                    "failure_has_object_texture": False
                }
            },
            
            "mock_rich_rag": {
                "context": (
                    "Warning! Past failures causes: \n"
                    "1. Misinterpretation: Grabbed the cylinder tomato can instead of the bottle.\n"
                    "2. Verbose Prompt: The policy failed to parse the rambling instruction.\n"
                    "[SUCCESS MEMORY] The policy strongly prefers concise, standardized instructions. "
                    "The 'bottle for fries' refers specifically to the red ketchup bottle. "
                    "To succeed, use the 'encore' tool to pass the image as-is, but strictly set the 'refined_task' parameter to: "
                    "'Pick the ketchup and place it in the basket' OR 'Pick the red sauce bottle and place it in the basket'."
                ),
                "hints": {
                    "success_has_object_texture": False, 
                    "failure_has_object_texture": False
                }
            }
        },
        
        "4_long_task_subtask": {
            "image": "/mnt/disk1/shared_data/lzy/SOMA/src/experiment_step/chain-step/data/original.jpg",  
            "task": "Sort the items: milk and cream cheese to the basket, tomato sauce to the plate.",   
            
            "mock_no_rag": {
                "context": "",
                "hints": {}
            },

            "mock_limited_rag": {
                "context": (
                    "Warning! Past failures causes: \n"
                    "1. Task Decomposition Error: The system failed to chunk the sequence correctly, either missing tasks or adding unnecessary hallucinated steps.\n"
                    "2. Semantic Misunderstanding: The system associated the wrong items with the wrong receptacles (e.g., trying to place milk on the plate instead of the basket)."
                ),
                "hints": {
                    "success_has_object_texture": False,
                    "failure_has_object_texture": False
                }
            },
            
            "mock_rich_rag": {
                "context": (
                    "Warning! Past failures causes: Task chunking errors and incorrect item-receptacle associations.\n"
                    "[SUCCESS MEMORY] The policy strictly requires atomic 'Pick [item] and place it in/on [receptacle]' subtasks to execute sequentially. "
                    "To succeed, you MUST use the 'task_decompose' tool and provide exactly these 3 subtasks in the 'subtasks' array: \n"
                    "1. 'Pick up the cream cheese and place it in the basket.' \n"
                    "2. 'Pick up the milk and place it in the basket.' \n"
                    "3. 'Pick up the tomato sauce and place it on the plate.'"
                ),
                "hints": {
                    "success_has_object_texture": False, 
                    "failure_has_object_texture": False
                }
            }
        }
    }
    
    final_report = {}

    for task_type, data in test_cases.items():
        if not data["image"]:
            logging.info(f"Skipping {task_type}, data not filled yet.")
            continue
            
        logging.info(f"=== Testing Category: {task_type} (Iterations: {iterations}) ===")
        
        category_scores = {
            "No_RAG": [], "Limited_RAG": [], "Rich_RAG": []
        }
        category_plans = {"No_RAG": [], "Limited_RAG": [], "Rich_RAG": []}

        for i in range(iterations):
            logging.info(f"  -> Run {i+1}/{iterations}...")
            
            # --- A. No RAG ---
            plan_a = vlm.orchestrate_perception(
                image=data["image"], 
                task_desc=data["task"], 
                rag_context=data["mock_no_rag"]["context"], 
                rag_hints=data["mock_no_rag"]["hints"]
            )
            score_a = evaluator.score_plan(task_type, data["image"], data["task"], plan_a, rag_context="") 
            category_scores["No_RAG"].append(score_a.get("Total", 0))
            category_plans["No_RAG"].append(plan_a)
            
            # --- B. Limited RAG ---
            plan_b = vlm.orchestrate_perception(
                image=data["image"], 
                task_desc=data["task"], 
                rag_context=data["mock_limited_rag"]["context"], 
                rag_hints=data["mock_limited_rag"]["hints"]
            )
            score_b = evaluator.score_plan(task_type, data["image"], data["task"], plan_b, rag_context=data["mock_limited_rag"]["context"])
            category_scores["Limited_RAG"].append(score_b.get("Total", 0))
            category_plans["Limited_RAG"].append(plan_b)

            # --- C. Rich RAG ---
            plan_c = vlm.orchestrate_perception(
                image=data["image"], 
                task_desc=data["task"], 
                rag_context=data["mock_rich_rag"]["context"], 
                rag_hints=data["mock_rich_rag"]["hints"]
            )
            score_c = evaluator.score_plan(task_type, data["image"], data["task"], plan_c, rag_context=data["mock_rich_rag"]["context"])
            category_scores["Rich_RAG"].append(score_c.get("Total", 0))
            category_plans["Rich_RAG"].append(plan_c)

        def safe_mean(scores_list):
            valid_scores = [s for s in scores_list if s >= 0]
            if not valid_scores:
                return 0.0
            return statistics.mean(valid_scores)
        
        # Calculate average scores and prepare final report for this category
        final_report[task_type] = {
            "Average_Scores": {
                "No_RAG": safe_mean(category_scores["No_RAG"]),
                "Limited_RAG": safe_mean(category_scores["Limited_RAG"]),
                "Rich_RAG": safe_mean(category_scores["Rich_RAG"])
            },
            "Raw_Scores": category_scores,
            "Sample_Plans_From_Last_Run": {
                "No_RAG": category_plans["No_RAG"][-1],
                "Limited_RAG": category_plans["Limited_RAG"][-1],
                "Rich_RAG": category_plans["Rich_RAG"][-1]
            }
        }
        
    with open("comprehensive_rag_eval.json", "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    logging.info("✅ All done! Results saved to comprehensive_rag_eval.json")

if __name__ == "__main__":
    # Defalt to 5 iterations for more robust averages
    run_comprehensive_ablation(iterations=5)