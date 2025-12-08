"""LLM-based reasoning module for multi-hop question answering."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLMReasoner:
    """Wrapper for open-source LLM to generate reasoning steps."""
    
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device=None):
        """
        Initialize LLM reasoner.
        
        Args:
            model_name: HuggingFace model name (default: Llama-3.1-8B-Instruct)
            device: torch device (auto-detected if None)
        """
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading LLM: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model - simpler approach without device_map
        print(f"Downloading/Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device in ["mps", "cuda"] else torch.float32,
            low_cpu_mem_usage=True
        )
        
        print(f"Moving model to {self.device}...")
        self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def reason(self, question, retrieved_docs, max_new_tokens=256, temperature=0.7):
        """
        Generate reasoning given question and retrieved documents.
        
        Args:
            question: Original question string
            retrieved_docs: List of retrieved document texts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated reasoning text (next sub-query or reasoning step)
        """
        # Build prompt
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""Given the following question and retrieved documents, generate a reasoning step or sub-query to find the answer.

Question: {question}

Retrieved Documents:
{context}

Based on the above information, what should be the next reasoning step or sub-query to answer the question? Generate a concise reasoning statement or a new search query.

Reasoning:"""

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after prompt)
        reasoning = generated_text[len(prompt):].strip()
        
        return reasoning
