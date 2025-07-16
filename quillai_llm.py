import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import re
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from collections import Counter

class QuillAILLM:
    def __init__(self, model_name="microsoft/DialoGPT-medium", force_model_check=True, debug_mode=False):
        """
        Initialize QuillAI LLM with specified model.
        
        Args:
            model_name: Hugging Face model identifier
            force_model_check: If True, validates model name and prevents problematic models
            debug_mode: If True, enables verbose debugging and token validation
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.debug_mode = debug_mode
        
        # NEW: Initialize semantic understanding for query analysis
        try:
            self.semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✓ Semantic understanding model loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load semantic model: {e}")
            self.semantic_model = None
        
        # NEW: Intent detection patterns
        self.intent_patterns = {
            'question_generation': [
                r'generate.*questions?', r'create.*questions?', r'make.*questions?',
                r'question.*paper', r'exam.*questions?', r'test.*questions?',
                r'quiz.*questions?', r'assessment.*questions?', r'\d+.*questions?'
            ],
            'rubric_creation': [
                r'rubric', r'marking.*scheme', r'grading.*criteria', 
                r'evaluation.*criteria', r'assessment.*criteria'
            ],
            'list_generation': [
                r'list.*of', r'enumerate', r'give.*examples?', r'provide.*examples?',
                r'name.*\d+', r'mention.*\d+', r'state.*\d+'
            ],
            'summary_request': [
                r'summarize', r'summary', r'key.*points?', r'main.*points?',
                r'overview', r'brief.*explanation'
            ],
            'comparison': [
                r'compare', r'contrast', r'difference', r'versus', r'vs\.?',
                r'similarities?.*differences?', r'pros.*cons'
            ],
            'definition': [
                r'what.*is', r'define', r'definition', r'meaning.*of',
                r'explain.*term', r'concept.*of'
            ],
            'explanation': [
                r'explain', r'how.*does', r'how.*to', r'describe',
                r'process.*of', r'steps?.*to', r'procedure'
            ],
            'application': [
                r'examples?.*of', r'applications?.*of', r'uses?.*of',
                r'real.*world', r'practical.*use', r'implement'
            ]
        }
        
        # NEW: Extended domain knowledge for multiple academic fields
        self.domain_knowledge = {
            'computer_science': {
                'algorithms': ['sorting', 'searching', 'graph', 'dynamic programming', 'greedy'],
                'machine_learning': ['supervised', 'unsupervised', 'neural networks', 'deep learning'],
                'data_structures': ['array', 'tree', 'graph', 'hash table', 'stack', 'queue'],
                'programming': ['object oriented', 'functional', 'procedural', 'languages']
            },
            'mathematics': {
                'calculus': ['derivative', 'integral', 'limit', 'continuity'],
                'linear_algebra': ['matrix', 'vector', 'eigenvalue', 'determinant'],
                'statistics': ['probability', 'distribution', 'hypothesis testing', 'regression'],
                'discrete_math': ['graph theory', 'combinatorics', 'logic', 'set theory']
            },
            'physics': {
                'mechanics': ['force', 'momentum', 'energy', 'motion'],
                'thermodynamics': ['entropy', 'enthalpy', 'heat', 'temperature'],
                'electromagnetism': ['electric field', 'magnetic field', 'current', 'voltage'],
                'quantum': ['superposition', 'entanglement', 'wave function', 'uncertainty']
            },
            'chemistry': {
                'organic': ['hydrocarbons', 'functional groups', 'reactions', 'synthesis'],
                'inorganic': ['periodic table', 'bonding', 'coordination', 'crystals'],
                'physical': ['thermodynamics', 'kinetics', 'equilibrium', 'spectroscopy'],
                'analytical': ['chromatography', 'spectroscopy', 'titration', 'mass spec']
            },
            'biology': {
                'molecular': ['dna', 'rna', 'proteins', 'enzymes', 'metabolism'],
                'cellular': ['cell structure', 'organelles', 'membrane', 'division'],
                'genetics': ['inheritance', 'mutations', 'gene expression', 'evolution'],
                'ecology': ['ecosystems', 'biodiversity', 'conservation', 'populations']
            },
            'economics': {
                'microeconomics': ['supply', 'demand', 'elasticity', 'market structures'],
                'macroeconomics': ['gdp', 'inflation', 'unemployment', 'monetary policy'],
                'finance': ['investment', 'risk', 'portfolio', 'derivatives'],
                'behavioral': ['decision making', 'biases', 'game theory', 'psychology']
            }
        }
        
        # Force check to ensure we're not loading problematic models
        if force_model_check and "gpt2-medium" in model_name.lower() and "microsoft" not in model_name.lower():
            raise ValueError(f"ERROR: Attempting to load base GPT-2 model '{model_name}'. Use 'microsoft/DialoGPT-medium' instead!")
        
        print(f"Loading model: {model_name} on {self.device} ...")
        print(f"Expected model: {model_name}")
        print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map=None  # We'll move to device manually
            )
            
            # Validate we loaded the correct model
            actual_model_name = getattr(self.model.config, '_name_or_path', model_name)
            print(f"Actual loaded model: {actual_model_name}")
            
            print("Model loaded successfully.")
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            if "gated repo" in str(e) or "access" in str(e).lower() or "401" in str(e):
                print(f"Error: Model {model_name} requires authentication or access.")
                print("Falling back to microsoft/DialoGPT-medium...")
                # Fallback to DialoGPT
                model_name = "microsoft/DialoGPT-medium"
                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    device_map=None
                )
                print(f"Successfully loaded fallback model: {model_name}")
                self.model.to(self.device)
                self.model.eval()
            else:
                raise e

        # Setup logging with current user context
        log_filename = f"quillai_llm_{datetime.utcnow().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s [AbhayRao38] %(levelname)s %(message)s",
            force=True
        )
        logging.info(f"Model {model_name} loaded on {self.device}")
        logging.info(f"User: AbhayRao38, Session start: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logging.info(f"Actual model path: {actual_model_name}")

    # NEW: Intent Detection System
    def detect_query_intent(self, query):
        """
        Detect the intent/type of the query using pattern matching and semantic analysis.
        
        Returns:
            tuple: (primary_intent, confidence_score, all_detected_intents)
        """
        query_lower = query.lower()
        detected_intents = {}
        
        # Pattern-based detection
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents[intent] = detected_intents.get(intent, 0) + 1
        
        # Semantic similarity detection (if available)
        if self.semantic_model:
            semantic_intents = self._semantic_intent_detection(query)
            for intent, score in semantic_intents.items():
                detected_intents[intent] = detected_intents.get(intent, 0) + score
        
        # Determine primary intent
        if detected_intents:
            primary_intent = max(detected_intents, key=detected_intents.get)
            confidence = detected_intents[primary_intent] / sum(detected_intents.values())
        else:
            primary_intent = 'definition'  # Default fallback
            confidence = 0.5
        
        return primary_intent, confidence, detected_intents

    def _semantic_intent_detection(self, query):
        """Use semantic similarity to detect query intent."""
        intent_examples = {
            'question_generation': "Generate 10 questions about machine learning",
            'rubric_creation': "Create a marking rubric for this assignment",
            'list_generation': "List 5 examples of sorting algorithms",
            'summary_request': "Summarize the key points of this topic",
            'comparison': "Compare supervised and unsupervised learning",
            'definition': "What is artificial intelligence?",
            'explanation': "Explain how neural networks work",
            'application': "Give examples of AI applications"
        }
        
        try:
            query_embedding = self.semantic_model.encode([query])
            intent_scores = {}
            
            for intent, example in intent_examples.items():
                example_embedding = self.semantic_model.encode([example])
                similarity = np.dot(query_embedding[0], example_embedding[0]) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(example_embedding[0])
                )
                if similarity > 0.3:  # Threshold for relevance
                    intent_scores[intent] = similarity * 2  # Weight semantic scores
            
            return intent_scores
        except Exception as e:
            if self.debug_mode:
                print(f"Semantic intent detection failed: {e}")
            return {}

    # NEW: Domain and Topic Detection
    def detect_domain_and_topic(self, query):
        """
        Detect the academic domain and specific topics from the query.
        
        Returns:
            tuple: (domain, topics, confidence)
        """
        query_lower = query.lower()
        domain_scores = {}
        detected_topics = []
        
        for domain, categories in self.domain_knowledge.items():
            domain_score = 0
            for category, topics in categories.items():
                for topic in topics:
                    if topic in query_lower:
                        domain_score += 1
                        detected_topics.append(topic)
            
            if domain_score > 0:
                domain_scores[domain] = domain_score
        
        # Determine primary domain
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[primary_domain] / sum(domain_scores.values())
        else:
            primary_domain = 'general'
            confidence = 0.3
        
        return primary_domain, detected_topics, confidence

    # NEW: Outline-Based Generation for Long Answers
    def generate_outline_then_expand(self, query, mode, marks, target_words):
        """
        Generate an outline first, then expand each section for comprehensive answers.
        """
        # Detect intent and domain
        intent, intent_confidence, _ = self.detect_query_intent(query)
        domain, topics, domain_confidence = self.detect_domain_and_topic(query)
        
        # Generate outline based on intent and domain
        outline = self._generate_outline(query, intent, domain, topics, target_words)
        
        # Expand each section
        expanded_sections = []
        words_per_section = target_words // len(outline) if outline else target_words
        
        for section_title, section_prompt in outline.items():
            section_content = self._expand_outline_section(
                section_title, section_prompt, words_per_section, domain
            )
            expanded_sections.append(f"**{section_title}**\n{section_content}")
        
        return "\n\n".join(expanded_sections)

    def _generate_outline(self, query, intent, domain, topics, target_words):
        """Generate a structured outline based on query characteristics."""
        outline = {}
        
        if intent == 'question_generation':
            outline = {
                "Question Types": "Identify appropriate question types for the topic",
                "Topic Coverage": "Ensure comprehensive coverage of key concepts",
                "Difficulty Levels": "Balance questions across different difficulty levels",
                "Sample Questions": "Generate specific questions with clear marking criteria"
            }
        elif intent == 'rubric_creation':
            outline = {
                "Assessment Criteria": "Define key evaluation criteria",
                "Marking Scheme": "Establish point allocation and grading scale",
                "Performance Levels": "Describe different levels of achievement",
                "Application Guidelines": "Provide guidance for consistent application"
            }
        elif intent == 'comparison':
            outline = {
                "Definition and Overview": "Define the concepts being compared",
                "Similarities": "Identify common characteristics and features",
                "Key Differences": "Highlight important distinctions",
                "Applications and Use Cases": "Compare practical applications"
            }
        elif target_words and target_words >= 400:  # Long answers
            if domain == 'computer_science':
                outline = {
                    "Definition and Fundamentals": "Core concepts and theoretical foundation",
                    "Technical Implementation": "How it works and key mechanisms",
                    "Practical Examples": "Real-world applications and case studies",
                    "Advantages and Limitations": "Benefits, challenges, and constraints",
                    "Future Directions": "Current trends and future developments"
                }
            elif domain == 'mathematics':
                outline = {
                    "Mathematical Definition": "Formal definition and notation",
                    "Key Properties": "Important mathematical properties and theorems",
                    "Solution Methods": "Techniques and approaches for problem-solving",
                    "Applications": "Real-world applications and examples",
                    "Related Concepts": "Connections to other mathematical areas"
                }
            else:
                outline = {
                    "Introduction and Definition": "Basic concepts and terminology",
                    "Key Characteristics": "Important features and properties",
                    "Examples and Applications": "Practical examples and use cases",
                    "Significance and Impact": "Importance and broader implications"
                }
        else:  # Standard outline for shorter answers
            outline = {
                "Definition": "Clear explanation of the concept",
                "Key Features": "Important characteristics and properties",
                "Examples": "Practical examples and applications"
            }
        
        return outline

    def _expand_outline_section(self, section_title, section_prompt, target_words, domain):
        """Expand a single outline section with domain-specific content."""
        # Use domain-specific templates and knowledge
        if domain in self.domain_knowledge:
            domain_terms = []
            for category, terms in self.domain_knowledge[domain].items():
                domain_terms.extend(terms)
            
            # Generate content with domain awareness
            content = self._generate_domain_specific_content(
                section_title, section_prompt, target_words, domain, domain_terms
            )
        else:
            # Generic content generation
            content = self._generate_generic_section_content(
                section_title, section_prompt, target_words
            )
        
        return content

    def _generate_domain_specific_content(self, section_title, section_prompt, target_words, domain, domain_terms):
        """Generate domain-specific content for outline sections."""
        content_templates = {
            'computer_science': {
                'Definition': "In computer science, this concept represents a fundamental principle that governs computational processes and algorithmic design.",
                'Technical Implementation': "The implementation involves systematic approaches using data structures, algorithms, and computational methods.",
                'Examples': "Common examples include applications in software development, system design, and algorithmic problem-solving.",
                'Applications': "This concept finds extensive use in software engineering, artificial intelligence, and system optimization."
            },
            'mathematics': {
                'Mathematical Definition': "Mathematically, this concept is defined through formal notation and rigorous mathematical frameworks.",
                'Key Properties': "Important properties include mathematical relationships, theorems, and fundamental principles.",
                'Solution Methods': "Solution approaches involve analytical techniques, computational methods, and mathematical reasoning.",
                'Applications': "Applications span across engineering, physics, computer science, and other quantitative fields."
            },
            'physics': {
                'Physical Principles': "The underlying physical principles involve fundamental laws and natural phenomena.",
                'Mathematical Framework': "The mathematical description uses equations, models, and quantitative relationships.",
                'Experimental Evidence': "Experimental observations and measurements support theoretical predictions.",
                'Applications': "Practical applications include technology development, engineering solutions, and scientific research."
            }
        }
        
        # Get template or use generic
        templates = content_templates.get(domain, content_templates['computer_science'])
        base_content = templates.get(section_title, "This section covers important aspects of the topic with detailed explanation and examples.")
        
        # Expand to target word count
        expanded_content = self._expand_content_to_target(base_content, target_words, domain_terms)
        
        return expanded_content

    def _expand_content_to_target(self, base_content, target_words, domain_terms):
        """Expand content to reach target word count using domain terms."""
        current_words = len(base_content.split())
        
        if current_words >= target_words:
            return base_content
        
        # Add domain-specific elaboration
        elaborations = [
            f"Key aspects include {', '.join(domain_terms[:3])} which demonstrate the comprehensive nature of this field.",
            "These concepts form the foundation for advanced understanding and practical application.",
            "Contemporary research continues to develop new methodologies and approaches.",
            "The interdisciplinary nature contributes to its broad applicability across various domains.",
            "Understanding these principles enables effective problem-solving and innovation."
        ]
        
        expanded = base_content
        for elaboration in elaborations:
            if len(expanded.split()) < target_words:
                expanded += f" {elaboration}"
            else:
                break
        
        return expanded

    # NEW: LLM Output Deduplication
    def deduplicate_response(self, text):
        """
        Remove repeated sentences and paragraphs from LLM output.
        
        Args:
            text: Input text that may contain repetitions
            
        Returns:
            str: Deduplicated text
        """
        if not text:
            return text
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Normalize sentence for comparison (remove extra spaces, convert to lowercase)
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            
            # Skip very short sentences (likely fragments)
            if len(normalized.split()) < 3:
                continue
            
            # Check for exact duplicates
            if normalized not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(normalized)
            else:
                if self.debug_mode:
                    print(f"Removed duplicate sentence: {sentence[:50]}...")
        
        # Reconstruct text
        deduplicated = '. '.join(unique_sentences)
        if deduplicated and not deduplicated.endswith('.'):
            deduplicated += '.'
        
        # Additional check for repeated phrases
        deduplicated = self._remove_repeated_phrases(deduplicated)
        
        return deduplicated

    def _remove_repeated_phrases(self, text):
        """Remove repeated phrases within the text."""
        words = text.split()
        
        # Look for repeated 3-5 word phrases
        for phrase_length in range(3, 6):
            phrase_counts = Counter()
            
            # Count phrase occurrences
            for i in range(len(words) - phrase_length + 1):
                phrase = ' '.join(words[i:i + phrase_length])
                phrase_counts[phrase] += 1
            
            # Remove repeated phrases (keep only first occurrence)
            for phrase, count in phrase_counts.items():
                if count > 1:
                    # Find all occurrences and remove extras
                    phrase_pattern = re.escape(phrase)
                    matches = list(re.finditer(phrase_pattern, text))
                    
                    if len(matches) > 1:
                        # Remove all but the first occurrence
                        for match in reversed(matches[1:]):
                            text = text[:match.start()] + text[match.end():]
                        
                        if self.debug_mode:
                            print(f"Removed repeated phrase: {phrase}")
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # ENHANCED: Generate Answer with New Features
    def generate_answer(
        self,
        query,
        mode="learning",
        marks=None,
        context_chunks=None,
        rerank_context=True,
        return_citations=True,
        feedback_callback=None,
        temperature=0.8,
        max_new_tokens=None
    ):
        """
        Enhanced generate_answer with intent detection, outline-expansion, and deduplication.
        """
        logs = []
        start_time = datetime.utcnow()
        
        # NEW: Detect query intent and domain
        intent, intent_confidence, all_intents = self.detect_query_intent(query)
        domain, topics, domain_confidence = self.detect_domain_and_topic(query)
        
        logs.append(f"Query: {query}")
        logs.append(f"Detected intent: {intent} (confidence: {intent_confidence:.2f})")
        logs.append(f"All intents: {all_intents}")
        logs.append(f"Detected domain: {domain} (confidence: {domain_confidence:.2f})")
        logs.append(f"Topics: {topics}")
        
        # Route to specialized handlers based on intent
        if intent == 'question_generation':
            return self._handle_question_generation(query, mode, marks, context_chunks, logs)
        elif intent == 'rubric_creation':
            return self._handle_rubric_creation(query, mode, marks, context_chunks, logs)
        elif intent == 'list_generation':
            return self._handle_list_generation(query, mode, marks, context_chunks, logs)
        elif intent == 'summary_request':
            return self._handle_summary_request(query, mode, marks, context_chunks, logs)
        
        # Continue with enhanced standard processing
        target_words = None
        if mode.lower() == "question" and marks is not None:
            target_words = {2: 100, 5: 250, 10: 500}.get(marks, 100)

        if max_new_tokens is None:
            if mode.lower() == "learning":
                max_new_tokens = 400
            else:
                if target_words:
                    max_new_tokens = min(max(target_words + 50, 150), 600)
                else:
                    max_new_tokens = 200

        max_model_tokens = 1024
        max_input_tokens = max_model_tokens - max_new_tokens - 30

        if max_input_tokens < 100:
            max_new_tokens = 200
            max_input_tokens = max_model_tokens - max_new_tokens - 30

        logs.append(f"Mode: {mode}, Marks: {marks}, Target words: {target_words}")
        logs.append(f"Temperature: {temperature}, Max new tokens: {max_new_tokens}")

        # Enhanced context handling with domain awareness
        selected_context = []
        if context_chunks and len(context_chunks) > 0:
            if rerank_context:
                context_chunks = self._enhanced_rerank_context(query, context_chunks, domain, topics)
            
            relevant_chunks = []
            for chunk in context_chunks[:2]:
                if self._is_context_relevant(chunk, query, domain, topics):
                    if len(chunk) > 300:
                        chunk = chunk[:300] + "..."
                    relevant_chunks.append(chunk)
            
            selected_context = relevant_chunks[:1]
            
            if selected_context:
                logs.append(f"Using {len(selected_context)} relevant context chunk(s)")
            else:
                logs.append("No relevant context found, proceeding without context")

        # NEW: Use outline-expansion for long answers
        if target_words and target_words >= 400:
            logs.append("Using outline-expansion method for long answer")
            answer = self.generate_outline_then_expand(query, mode, marks, target_words)
        else:
            # Create enhanced prompt with domain and intent awareness
            prompt = self._create_domain_aware_prompt(query, mode, marks, target_words, selected_context, intent, domain, topics)
            
            # Continue with LLM generation
            answer = self._generate_with_llm(prompt, max_input_tokens, max_new_tokens, temperature, query, mode, marks, target_words, selected_context, return_citations, logs)
        
        # NEW: Apply deduplication
        answer = self.deduplicate_response(answer)
        logs.append("Applied deduplication to remove repeated content")
        
        # Enhanced structure and expansion
        answer = self._enhance_and_expand_response(answer, query, mode, target_words)
        
        # Ensure target word count for question mode
        if mode.lower() == "question" and target_words:
            current_words = len(answer.split())
            if current_words < target_words * 0.7:
                answer = self._expand_to_target_words(answer, query, target_words, mode)
                logs.append(f"Expanded response to meet target word count")

        # Add citations if requested and we used relevant context
        if return_citations and selected_context:
            citations = []
            for i, chunk in enumerate(selected_context):
                clean_chunk = re.sub(r'[\s\n\r]+', ' ', chunk[:80])
                if any(term in clean_chunk.lower() for term in ['algorithm', 'machine', 'computer', 'data', 'learning']):
                    citations.append(f"[{i+1}] {clean_chunk}")
            
            if citations:
                answer += "\n\nReferences:\n" + "\n".join(citations)

        # Final quality check and cleanup
        answer = self._final_quality_check(answer, query, mode, target_words)
        logs.append(f"Final answer length: {len(answer)} chars, words: {len(answer.split())}")
        logs.append(f"Total processing time: {(datetime.utcnow() - start_time).total_seconds():.2f} seconds")

        self._write_logs(logs)

        if feedback_callback is not None:
            feedback_callback(query, answer, context_chunks)

        return answer

    # NEW: Specialized Intent Handlers
    def _handle_question_generation(self, query, mode, marks, context_chunks, logs):
        """Handle question generation requests."""
        logs.append("Routing to question generation handler")
        
        # Extract number of questions
        numbers = re.findall(r'\d+', query)
        num_questions = int(numbers[0]) if numbers else 5
        num_questions = min(num_questions, 15)  # Cap at 15
        
        # Extract topic
        topic = self._extract_topic_from_query(query)
        
        # Generate questions based on marks/difficulty
        if marks == 2:
            question_types = ['definition', 'short_answer']
        elif marks == 5:
            question_types = ['explanation', 'comparison', 'application']
        elif marks == 10:
            question_types = ['analysis', 'evaluation', 'synthesis']
        else:
            question_types = ['definition', 'explanation', 'application']
        
        questions = self._generate_topic_questions(topic, num_questions, question_types)
        
        # Format response
        response = f"**Question Paper: {topic.title()}**\n\n"
        if marks:
            response += f"*Instructions: Each question carries {marks} marks.*\n\n"
        
        for i, question in enumerate(questions, 1):
            response += f"**Q{i}.** {question}\n\n"
        
        response += f"*Total Questions: {num_questions}*\n"
        response += f"*Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*"
        
        return response

    def _handle_rubric_creation(self, query, mode, marks, context_chunks, logs):
        """Handle rubric creation requests."""
        logs.append("Routing to rubric creation handler")
        
        topic = self._extract_topic_from_query(query)
        marks = marks or 10  # Default to 10 marks
        
        rubric = f"**Marking Rubric: {topic.title()}**\n\n"
        rubric += f"*Total Marks: {marks}*\n\n"
        
        # Create rubric based on marks
        if marks <= 2:
            criteria = [
                ("Definition/Understanding", marks * 0.6),
                ("Clarity/Expression", marks * 0.4)
            ]
        elif marks <= 5:
            criteria = [
                ("Conceptual Understanding", marks * 0.4),
                ("Examples/Applications", marks * 0.3),
                ("Clarity and Structure", marks * 0.3)
            ]
        else:
            criteria = [
                ("Theoretical Understanding", marks * 0.3),
                ("Practical Examples", marks * 0.25),
                ("Analysis and Evaluation", marks * 0.25),
                ("Structure and Presentation", marks * 0.2)
            ]
        
        for criterion, mark_allocation in criteria:
            rubric += f"**{criterion}** ({mark_allocation:.1f} marks)\n"
            rubric += f"- Excellent: Clear, comprehensive, accurate\n"
            rubric += f"- Good: Generally accurate with minor gaps\n"
            rubric += f"- Fair: Basic understanding with some errors\n"
            rubric += f"- Poor: Significant gaps or inaccuracies\n\n"
        
        return rubric

    def _handle_list_generation(self, query, mode, marks, context_chunks, logs):
        """Handle list generation requests."""
        logs.append("Routing to list generation handler")
        
        # Extract number and topic
        numbers = re.findall(r'\d+', query)
        num_items = int(numbers[0]) if numbers else 5
        num_items = min(num_items, 10)  # Cap at 10
        
        topic = self._extract_topic_from_query(query)
        
        # Generate list items
        items = self._generate_topic_list(topic, num_items, query)
        
        response = f"**{num_items} {topic.title()} Examples:**\n\n"
        for i, item in enumerate(items, 1):
            response += f"{i}. **{item['name']}**: {item['description']}\n\n"
        
        return response

    def _handle_summary_request(self, query, mode, marks, context_chunks, logs):
        """Handle summary requests."""
        logs.append("Routing to summary handler")
        
        if not context_chunks:
            return "No content available to summarize. Please provide context or specify the topic to summarize."
        
        # Combine context
        combined_text = " ".join(context_chunks)
        
        # Extract key points
        key_points = self._extract_key_points(combined_text)
        
        # Generate summary
        summary = f"**Summary of Key Points:**\n\n"
        for i, point in enumerate(key_points[:5], 1):
            summary += f"{i}. {point}\n\n"
        
        summary += f"**Overview:** This content covers {len(key_points)} main concepts that are essential for understanding the topic."
        
        return summary

    # NEW: Enhanced Context Reranking with Domain Awareness
    def _enhanced_rerank_context(self, query, context_chunks, domain, topics):
        """Enhanced context reranking with domain awareness."""
        def score_chunk(chunk):
            chunk_lower = chunk.lower()
            query_lower = query.lower()
            
            # Base score from query word matches
            query_words = set(query_lower.split())
            chunk_words = set(chunk_lower.split())
            base_score = len(query_words & chunk_words) * 10
            
            # Domain-specific bonus
            domain_bonus = 0
            if domain in self.domain_knowledge:
                domain_terms = []
                for category, terms in self.domain_knowledge[domain].items():
                    domain_terms.extend(terms)
                domain_bonus = sum(5 for term in domain_terms if term in chunk_lower)
            
            # Topic-specific bonus
            topic_bonus = sum(8 for topic in topics if topic in chunk_lower)
            
            # Academic content bonus
            academic_terms = ['definition', 'example', 'characteristic', 'principle', 'method', 'approach']
            academic_bonus = sum(3 for term in academic_terms if term in chunk_lower)
            
            # Irrelevant content penalty
            irrelevant_terms = {'steve martin', 'millionaire', 'caveat lector', 'bibliography'}
            if any(term in chunk_lower for term in irrelevant_terms):
                return -1000
            
            total_score = base_score + domain_bonus + topic_bonus + academic_bonus
            return total_score
        
        ranked_chunks = sorted(context_chunks, key=score_chunk, reverse=True)
        return [chunk for chunk in ranked_chunks if score_chunk(chunk) > 0]

    def _is_context_relevant(self, chunk, query, domain, topics):
        """Check if context chunk is relevant to query with domain awareness."""
        chunk_lower = chunk.lower()
        query_lower = query.lower()
        
        # Check for irrelevant content
        irrelevant_terms = {'steve martin', 'millionaire', 'caveat lector', 'bibliography'}
        if any(term in chunk_lower for term in irrelevant_terms):
            return False
        
        # Check query word overlap
        query_words = set(query_lower.split())
        chunk_words = set(chunk_lower.split())
        overlap = len(query_words & chunk_words)
        
        if overlap > 0:
            return True
        
        # Check domain-specific terms
        if domain in self.domain_knowledge:
            domain_terms = []
            for category, terms in self.domain_knowledge[domain].items():
                domain_terms.extend(terms)
            if any(term in chunk_lower for term in domain_terms):
                return True
        
        # Check for academic content
        academic_indicators = ['algorithm', 'method', 'approach', 'technique', 'process', 'system', 'model', 'theory', 'principle', 'concept']
        return any(indicator in chunk_lower for indicator in academic_indicators)

    # NEW: Domain-Aware Prompt Creation
    def _create_domain_aware_prompt(self, query, mode, marks, target_words, context_chunks, intent, domain, topics):
        """Create domain-aware prompt with intent-specific instructions."""
        parts = []
        
        # Add relevant context if available
        if context_chunks:
            context_text = context_chunks[0]
            if len(context_text) > 200:
                context_text = context_text[:200] + "..."
            parts.append(f"Context: {context_text}")
        
        # Create intent and domain-specific instructions
        if intent == 'definition':
            instruction = self._get_definition_instruction(mode, target_words, domain)
        elif intent == 'explanation':
            instruction = self._get_explanation_instruction(mode, target_words, domain)
        elif intent == 'comparison':
            instruction = self._get_comparison_instruction(mode, target_words, domain)
        elif intent == 'application':
            instruction = self._get_application_instruction(mode, target_words, domain)
        else:
            instruction = self._get_general_instruction(mode, target_words, domain)
        
        parts.append(f"Instructions: {instruction}")
        parts.append(f"Question: {query}")
        parts.append("Academic Response:")
        
        return "\n\n".join(parts)

    def _get_definition_instruction(self, mode, target_words, domain):
        """Get definition-specific instruction."""
        base = "Provide a clear, comprehensive definition with key characteristics and examples."
        
        if domain == 'computer_science':
            base += " Include technical details and algorithmic aspects where relevant."
        elif domain == 'mathematics':
            base += " Include mathematical formulation and properties where applicable."
        elif domain == 'physics':
            base += " Include physical principles and real-world phenomena."
        
        if target_words:
            base += f" Write exactly {target_words} words in formal academic tone."
        else:
            base += " Write 200-300 words in formal academic tone."
        
        return base

    def _get_explanation_instruction(self, mode, target_words, domain):
        """Get explanation-specific instruction."""
        base = "Provide a detailed explanation of the process, mechanism, or concept."
        
        if domain == 'computer_science':
            base += " Include step-by-step algorithmic details and implementation considerations."
        elif domain == 'mathematics':
            base += " Include mathematical derivations and proof techniques where relevant."
        elif domain == 'physics':
            base += " Include underlying physical principles and mathematical relationships."
        
        if target_words:
            base += f" Structure your response clearly and write exactly {target_words} words."
        else:
            base += " Structure your response clearly with examples and write 250-350 words."
        
        return base

    def _get_comparison_instruction(self, mode, target_words, domain):
        """Get comparison-specific instruction."""
        base = "Compare and contrast the concepts, highlighting similarities, differences, and relative advantages."
        
        if domain == 'computer_science':
            base += " Include performance comparisons, complexity analysis, and use cases."
        elif domain == 'mathematics':
            base += " Include mathematical properties and applicability in different contexts."
        elif domain == 'physics':
            base += " Include physical principles and practical applications."
        
        if target_words:
            base += f" Organize your comparison systematically and write exactly {target_words} words."
        else:
            base += " Organize your comparison systematically and write 300-400 words."
        
        return base

    def _get_application_instruction(self, mode, target_words, domain):
        """Get application-specific instruction."""
        base = "Discuss practical applications and real-world uses with specific examples."
        
        if domain == 'computer_science':
            base += " Include industry applications, software implementations, and emerging technologies."
        elif domain == 'mathematics':
            base += " Include applications in science, engineering, and technology."
        elif domain == 'physics':
            base += " Include technological applications and engineering implementations."
        
        if target_words:
            base += f" Provide concrete examples and write exactly {target_words} words."
        else:
            base += " Provide concrete examples and write 250-350 words."
        
        return base

    def _get_general_instruction(self, mode, target_words, domain):
        """Get general instruction for other intents."""
        if mode.lower() == "learning":
            instruction = (
                "Provide a comprehensive academic explanation. "
                "Include: 1) Clear definition, 2) Key characteristics/principles, "
                "3) Multiple practical examples, 4) Applications and significance. "
                "Write 200-300 words in formal academic tone."
            )
        else:
            if target_words:
                instruction = (
                    f"Provide a complete academic answer in approximately {target_words} words. "
                    f"Include: 1) Clear definition, 2) Main characteristics, 3) Examples. "
                    f"Write exactly {target_words} words in formal academic tone."
                )
            else:
                instruction = (
                    "Provide a concise academic answer with definition, key points, and examples. "
                    "Write 100-150 words in formal academic tone."
                )
        
        return instruction

    # Helper methods for new functionality
    def _generate_topic_questions(self, topic, num_questions, question_types):
        """Generate topic-specific questions."""
        questions = []
        
        # Question templates by type
        templates = {
            'definition': [
                f"Define {topic} and explain its key characteristics.",
                f"What is {topic}? Provide a comprehensive definition.",
                f"Explain the concept of {topic} with suitable examples."
            ],
            'short_answer': [
                f"List the main features of {topic}.",
                f"What are the advantages of {topic}?",
                f"Briefly explain the importance of {topic}."
            ],
            'explanation': [
                f"Explain how {topic} works with detailed examples.",
                f"Describe the process involved in {topic}.",
                f"Explain the working principle of {topic}."
            ],
            'comparison': [
                f"Compare different types of {topic}.",
                f"What are the similarities and differences between various {topic} approaches?",
                f"Contrast the advantages and disadvantages of {topic}."
            ],
            'application': [
                f"Discuss the real-world applications of {topic}.",
                f"How is {topic} used in modern technology?",
                f"Provide examples of {topic} in practical scenarios."
            ],
            'analysis': [
                f"Analyze the impact of {topic} on modern computing.",
                f"Critically evaluate the effectiveness of {topic}.",
                f"Examine the challenges and limitations of {topic}."
            ],
            'evaluation': [
                f"Evaluate the significance of {topic} in computer science.",
                f"Assess the future prospects of {topic}.",
                f"Critically analyze the role of {topic} in solving real-world problems."
            ],
            'synthesis': [
                f"Design a system that incorporates {topic} principles.",
                f"Propose improvements to existing {topic} methods.",
                f"Synthesize information about {topic} to solve a complex problem."
            ]
        }
        
        # Generate questions using templates
        question_count = 0
        for question_type in question_types:
            if question_count >= num_questions:
                break
            
            type_templates = templates.get(question_type, templates['definition'])
            questions_needed = min(num_questions - question_count, len(type_templates))
            
            for i in range(questions_needed):
                if i < len(type_templates):
                    questions.append(type_templates[i])
                    question_count += 1
        
        # Fill remaining slots with mixed questions
        while len(questions) < num_questions:
            remaining_types = [t for t in templates.keys() if t in question_types]
            if remaining_types:
                question_type = remaining_types[len(questions) % len(remaining_types)]
                type_templates = templates[question_type]
                template_idx = len(questions) % len(type_templates)
                questions.append(type_templates[template_idx])
        
        return questions[:num_questions]

    def _generate_topic_list(self, topic, num_items, query):
        """Generate a list of items related to the topic."""
        # This would be expanded with more sophisticated list generation
        items = []
        
        # Basic template-based list generation
        for i in range(num_items):
            items.append({
                'name': f"{topic.title()} Example {i+1}",
                'description': f"This is an important example of {topic} that demonstrates key concepts and practical applications."
            })
        
        return items

    def _extract_key_points(self, text):
        """Extract key points from text for summarization."""
        sentences = text.split('.')
        key_points = []
        
        # Simple heuristic: look for sentences with key indicators
        key_indicators = ['important', 'key', 'main', 'primary', 'essential', 'fundamental', 'significant', 'crucial']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum length
                if any(indicator in sentence.lower() for indicator in key_indicators):
                    key_points.append(sentence)
                elif sentence.count(',') >= 2:  # Complex sentences often contain key info
                    key_points.append(sentence)
        
        return key_points[:10]  # Limit to top 10

    def _generate_with_llm(self, prompt, max_input_tokens, max_new_tokens, temperature, query, mode, marks, target_words, selected_context, return_citations, logs):
        """Generate response using the LLM (keeping existing implementation)."""
        # Tokenize and ensure within limits
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        original_length = len(prompt_ids)
        
        # Truncation if needed
        if len(prompt_ids) > max_input_tokens:
            if selected_context:
                prompt = self._create_domain_aware_prompt(query, mode, marks, target_words, [], 'definition', 'general', [])
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                logs.append("Removed context due to length")
            
            if len(prompt_ids) > max_input_tokens:
                excess = len(prompt_ids) - max_input_tokens
                prompt_ids = prompt_ids[excess:]
                logs.append(f"Truncated {excess} tokens from prompt")

        logs.append(f"Final input token length: {len(prompt_ids)} / {max_input_tokens}")

        # Prepare tensors
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        try:
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=30,
                    temperature=temperature,
                    top_p=0.92,
                    top_k=50,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3
                )

            # Decode and clean response
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self._extract_and_clean_response(result, prompt, prompt_ids)
            
            return answer

        except Exception as e:
            logs.append(f"Exception occurred: {repr(e)}")
            logging.error(f"Generation error for user AbhayRao38: {repr(e)}")
            
            # Fallback
            return self._generate_expanded_response(query, mode, marks, target_words, selected_context)

    # Keep all existing methods from your original file...
    # (I'm preserving your existing implementation for the methods below)

    def _create_enhanced_prompt(self, query, mode, marks, target_words, context_chunks):
        """Create an enhanced prompt that encourages proper response length and structure."""
        
        parts = []
        
        # Add relevant context if available
        if context_chunks:
            context_text = context_chunks[0]
            if len(context_text) > 200:
                context_text = context_text[:200] + "..."
            parts.append(f"Context: {context_text}")
        
        # Create mode-specific instruction with word count guidance
        if mode.lower() == "learning":
            instruction = (
                "Provide a comprehensive academic explanation. "
                "Include: 1) Clear definition, 2) Key characteristics/principles, "
                "3) Multiple practical examples, 4) Applications and significance. "
                "Write 200-300 words in formal academic tone."
            )
        else:
            if target_words:
                instruction = (
                    f"Provide a complete academic answer in approximately {target_words} words. "
                    f"Include: 1) Clear definition, 2) Main characteristics, 3) Examples. "
                    f"Write exactly {target_words} words in formal academic tone. Be comprehensive within the word limit."
                )
            else:
                instruction = (
                    "Provide a concise academic answer with definition, key points, and examples. "
                    "Write 100-150 words in formal academic tone."
                )
        
        parts.append(f"Instructions: {instruction}")
        parts.append(f"Question: {query}")
        parts.append("Academic Response:")
        
        return "\n\n".join(parts)

    def _extract_and_clean_response(self, result, prompt, prompt_ids):
        """Extract and clean the model's response from the full output."""
        
        # Extract only the model's response (remove prompt)
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        if prompt_text in result:
            answer = result.split(prompt_text, 1)[-1].strip()
        else:
            # Fallback: try to find "Academic Response:" marker
            if "Academic Response:" in result:
                answer = result.split("Academic Response:", 1)[-1].strip()
            else:
                answer = result.strip()

        # Clean up DialoGPT output issues
        answer = re.sub(r'<\|endoftext\|>.*$', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'<pad>.*$', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'<unk>.*$', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)  # Clean up excessive newlines

        # Remove repetitive patterns but be less aggressive
        lines = answer.split('\n')
        cleaned_lines = []
        prev_line = ""
        for line in lines:
            line = line.strip()
            if line and (line != prev_line or len(line) < 30):  # Allow some repetition for structure
                cleaned_lines.append(line)
                prev_line = line

        return '\n'.join(cleaned_lines)

    def _generate_expanded_response(self, query, mode, marks, target_words, context_chunks):
        """Generate a comprehensive response for the query with proper length."""
        
        query_lower = query.lower()
        
        # Enhanced keyword detection using more flexible matching
        if any(term in query_lower for term in ['machine learning', 'ml', 'supervised learning', 'unsupervised learning', 'deep learning']):
            return self._generate_ml_response(mode, target_words)
        elif any(term in query_lower for term in ['neural network', 'neural', 'deep learning', 'cnn', 'rnn', 'layers']):
            return self._generate_neural_response(mode, target_words)
        elif any(term in query_lower for term in ['algorithm', 'algorithms', 'sorting', 'searching', 'complexity']):
            return self._generate_algorithm_response(mode, target_words)
        elif any(term in query_lower for term in ['artificial intelligence', 'ai', 'expert system', 'intelligent']):
            return self._generate_ai_response(mode, target_words)
        elif any(term in query_lower for term in ['data structure', 'data structures', 'array', 'tree', 'graph']):
            return self._generate_data_structure_response(mode, target_words)
        elif any(term in query_lower for term in ['programming', 'coding', 'software', 'development']):
            return self._generate_programming_response(mode, target_words)
        elif any(term in query_lower for term in ['quantum', 'quantum computing', 'qubit', 'superposition']):
            return self._generate_quantum_response(mode, target_words)
        else:
            return self._generate_generic_response(query, mode, target_words)

    def _generate_ml_response(self, mode, target_words):
        """Generate a comprehensive machine learning response."""
        
        base_response = {
            'definition': "Machine learning is a subset of artificial intelligence (AI) that enables computer systems to automatically learn and improve from experience without being explicitly programmed for every task.",
            'characteristics': [
                "Data-driven learning and pattern recognition capabilities",
                "Automatic improvement of performance through iterative training",
                "Ability to make predictions or decisions on new, unseen data",
                "Statistical and mathematical modeling of complex relationships",
                "Adaptive algorithms that can generalize from training examples"
            ],
            'examples': [
                "Supervised learning: Email spam detection, medical diagnosis systems, credit score assessment",
                "Unsupervised learning: Customer segmentation, anomaly detection, data clustering",
                "Reinforcement learning: Game-playing AI (AlphaGo), autonomous vehicles, trading algorithms",
                "Deep learning: Image recognition, natural language processing, speech recognition"
            ],
            'applications': [
                "Healthcare: Medical imaging analysis, drug discovery, personalized treatment",
                "Finance: Fraud detection, algorithmic trading, risk assessment",
                "Technology: Recommendation systems, search engines, virtual assistants",
                "Transportation: Autonomous vehicles, route optimization, traffic management"
            ]
        }
        
        return self._format_academic_response(base_response, mode, target_words, "Machine Learning")

    def _generate_algorithm_response(self, mode, target_words):
        """Generate a comprehensive algorithms response."""
        
        base_response = {
            'definition': "An algorithm is a finite, well-defined sequence of computational instructions designed to solve a specific problem or perform a particular task systematically and efficiently.",
            'characteristics': [
                "Finite number of steps with clear termination conditions",
                "Unambiguous instructions with precise input and output specifications",
                "Effectiveness in solving the intended problem within reasonable time",
                "Deterministic behavior producing consistent results for identical inputs",
                "Computational efficiency considering time and space complexity"
            ],
            'examples': [
                "Sorting algorithms: Quicksort, mergesort, heapsort for data organization",
                "Search algorithms: Binary search, depth-first search, breadth-first search",
                "Graph algorithms: Dijkstra's shortest path, minimum spanning tree algorithms",
                "Dynamic programming: Fibonacci sequence, knapsack problem solutions"
            ],
            'applications': [
                "Database systems: Query optimization, indexing, data retrieval",
                "Computer graphics: Rendering algorithms, image processing, animation",
                "Network systems: Routing protocols, load balancing, data compression",
                "Software engineering: Compiler optimization, memory management, scheduling"
            ]
        }
        
        return self._format_academic_response(base_response, mode, target_words, "Algorithms")

    def _generate_neural_response(self, mode, target_words):
        """Generate a comprehensive neural networks response."""
        
        base_response = {
            'definition': "Neural networks are computational models inspired by biological neural systems, consisting of interconnected processing units (neurons) that learn to recognize patterns and make decisions through weighted connections.",
            'characteristics': [
                "Layered architecture with input, hidden, and output layers",
                "Weighted connections between neurons that adjust during training",
                "Activation functions that determine neuron output based on inputs",
                "Backpropagation algorithm for learning from training data",
                "Parallel processing capabilities for complex pattern recognition"
            ],
            'examples': [
                "Feedforward networks: Multilayer perceptrons for classification tasks",
                "Convolutional networks: Image recognition, computer vision applications",
                "Recurrent networks: Natural language processing, sequence prediction",
                "Deep networks: Complex feature learning, representation learning"
            ],
            'applications': [
                "Computer vision: Image classification, object detection, facial recognition",
                "Natural language: Machine translation, sentiment analysis, chatbots",
                "Healthcare: Medical image analysis, drug discovery, diagnostic systems",
                "Autonomous systems: Self-driving cars, robotics, game playing"
            ]
        }
        
        return self._format_academic_response(base_response, mode, target_words, "Neural Networks")

    def _generate_ai_response(self, mode, target_words):
        """Generate a comprehensive AI response."""
        
        base_response = {
            'definition': "Artificial Intelligence (AI) is the field of computer science focused on creating systems capable of performing tasks that typically require human intelligence, including reasoning, learning, and problem-solving.",
            'characteristics': [
                "Cognitive capabilities including reasoning and decision-making",
                "Learning and adaptation from experience and data",
                "Natural language understanding and communication",
                "Pattern recognition and sensory perception",
                "Problem-solving and strategic thinking abilities"
            ],
            'examples': [
                "Expert systems: Medical diagnosis, financial planning, technical support",
                "Machine learning: Predictive analytics, recommendation systems, data mining",
                "Computer vision: Image recognition, autonomous navigation, quality control",
                "Natural language processing: Chatbots, translation services, text analysis"
            ],
            'applications': [
                "Healthcare: Diagnostic assistance, treatment planning, medical research",
                "Education: Personalized learning, intelligent tutoring systems, assessment",
                "Business: Process automation, customer service, market analysis",
                "Research: Scientific discovery, data analysis, hypothesis generation"
            ]
        }
        
        return self._format_academic_response(base_response, mode, target_words, "Artificial Intelligence")

    def _generate_data_structure_response(self, mode, target_words):
        """Generate a comprehensive data structures response."""
        
        base_response = {
            'definition': "Data structures are specialized formats for organizing, storing, and managing data in computer systems to enable efficient access, modification, and processing operations.",
            'characteristics': [
                "Systematic organization of data elements and their relationships",
                "Defined operations for insertion, deletion, and retrieval",
                "Memory efficiency and optimal space utilization",
                "Time complexity considerations for various operations",
                "Abstract data types with specific behavioral properties"
            ],
            'examples': [
                "Linear structures: Arrays, linked lists, stacks, queues",
                "Tree structures: Binary trees, AVL trees, B-trees, heaps",
                "Graph structures: Directed graphs, undirected graphs, weighted graphs",
                "Hash structures: Hash tables, dictionaries, associative arrays"
            ],
            'applications': [
                "Database systems: Indexing, query processing, data organization",
                "Operating systems: Memory management, file systems, process scheduling",
                "Compiler design: Symbol tables, syntax trees, optimization",
                "Network algorithms: Routing tables, topology representation, path finding"
            ]
        }
        
        return self._format_academic_response(base_response, mode, target_words, "Data Structures")

    def _generate_programming_response(self, mode, target_words):
        """Generate a comprehensive programming response."""
        
        base_response = {
            'definition': "Programming is the process of designing, writing, testing, and maintaining computer programs using programming languages to instruct computers to perform specific tasks or solve computational problems.",
            'characteristics': [
                "Logical problem decomposition and algorithmic thinking",
                "Syntax adherence to programming language specifications",
                "Code organization through functions, classes, and modules",
                "Debugging and testing for correctness and reliability",
                "Documentation and maintainability considerations"
            ],
            'examples': [
                "System programming: Operating systems, device drivers, embedded systems",
                "Application development: Desktop software, mobile apps, web applications",
                "Data analysis: Statistical computing, data visualization, machine learning",
                "Game development: Graphics engines, game logic, user interfaces"
            ],
            'applications': [
                "Software industry: Commercial applications, enterprise systems, tools",
                "Scientific computing: Simulation, modeling, data analysis",
                "Web development: Websites, web services, e-commerce platforms",
                "Automation: Scripting, workflow automation, system administration"
            ]
        }
        
        return self._format_academic_response(base_response, mode, target_words, "Programming")

    def _generate_quantum_response(self, mode, target_words):
        """Generate a comprehensive quantum computing response."""
        
        base_response = {
            'definition': "Quantum computing is a revolutionary computational paradigm that leverages quantum mechanical phenomena such as superposition and entanglement to process information in fundamentally different ways than classical computers.",
            'characteristics': [
                "Quantum bits (qubits) that can exist in superposition states",
                "Quantum entanglement enabling correlated qubit behaviors",
                "Quantum interference for computational advantage",
                "Exponential scaling potential for certain problem types",
                "Quantum gates and circuits for information processing"
            ],
            'examples': [
                "Quantum algorithms: Shor's algorithm for factoring, Grover's search algorithm",
                "Quantum simulation: Molecular modeling, materials science applications",
                "Quantum cryptography: Quantum key distribution, secure communications",
                "Quantum machine learning: Quantum neural networks, optimization problems"
            ],
            'applications': [
                "Cryptography: Breaking classical encryption, quantum-safe security",
                "Drug discovery: Molecular simulation, protein folding prediction",
                "Finance: Portfolio optimization, risk analysis, fraud detection",
                "Artificial intelligence: Enhanced machine learning, pattern recognition"
            ]
        }
        
        return self._format_academic_response(base_response, mode, target_words, "Quantum Computing")

    def _generate_generic_response(self, query, mode, target_words):
        """Generate a generic academic response for unknown topics."""
        
        # FIXED: Improved topic extraction and sanitization
        topic = query.replace("What is ", "").replace("Define ", "").replace("Explain ", "").strip()
        topic = re.sub(r"[^a-zA-Z0-9\s]", "", topic).strip()
        
        # Remove conversational fluff
        topic = re.sub(r"\b(hi|hello|please|thanks|thank you|can you|tell me about|like|you know|those|something|stuff in)\b", "", topic, flags=re.IGNORECASE).strip()
        
        # Clean up multiple spaces and title case
        topic = re.sub(r'\s+', ' ', topic).strip()
        
        # Length guard - prevent absurdly long topics
        if len(topic.split()) > 8:
            topic = " ".join(topic.split()[:6])
        
        # Fallback if topic becomes empty
        if not topic or len(topic) < 3 or topic.lower() in ['ai', 'it', 'that', 'this']:
            topic = "Academic Topic"
        else:
            topic = topic.title()
        
        if mode.lower() == "learning":
            newline = "\n"
            response = f"""**Academic Explanation: {topic}**

**Definition:**
{topic} is a specialized area of study that encompasses specific principles, methodologies, and applications within its domain. Understanding this concept requires systematic examination of its fundamental components and practical implementations.

**Key Characteristics:**
The field is characterized by established theoretical frameworks, empirical methodologies, and practical applications. Key aspects include systematic approaches to problem-solving, evidence-based practices, and continuous evolution through research and development.

**Practical Examples:**
Applications of {topic.lower()} can be found across various domains, including academic research, industry practices, and technological implementations. Specific examples depend on the context and scope of application within the field.

**Significance:**
Understanding {topic.lower()} is essential for developing expertise in related areas and contributes to broader knowledge advancement in the field. This knowledge forms the foundation for further specialization and practical application.

**Conclusion:**
Mastery of {topic.lower()} requires both theoretical understanding and practical experience, making it a valuable area of study for students and professionals in related fields."""
        else:
            target_words = target_words or 100
            if target_words <= 100:
                response = f"{topic} is a specialized area of study with established principles and methodologies. Key characteristics include systematic approaches and evidence-based practices. Applications are found across academic and industry contexts, making it valuable for professional development."
            elif target_words <= 250:
                newline = "\n"
                response = f"""**Definition:** {topic} is a specialized area of study that encompasses specific principles, methodologies, and applications within its domain.{newline}{newline}**Key Characteristics:** The field is characterized by established theoretical frameworks, empirical methodologies, and practical applications across various contexts.{newline}{newline}**Examples:** Applications include academic research, industry practices, and technological implementations, demonstrating its broad relevance and practical value in professional settings."""
            else:
                newline = "\n"
                response = f"""**Definition:** {topic} is a specialized area of study that encompasses specific principles, methodologies, and applications within its domain. Understanding this concept requires systematic examination of its fundamental components.{newline}{newline}**Key Characteristics:** The field is characterized by established theoretical frameworks, empirical methodologies, and practical applications. Key aspects include systematic approaches to problem-solving, evidence-based practices, and continuous evolution through research and development.{newline}{newline}**Examples:** Applications of {topic.lower()} can be found across various domains, including academic research, industry practices, and technological implementations. Specific examples demonstrate the practical value and broad applicability of this knowledge.{newline}{newline}**Significance:** Understanding {topic.lower()} is essential for developing expertise in related areas and contributes to broader knowledge advancement, making it valuable for both students and professionals."""
        
        return response

    def _format_academic_response(self, content_dict, mode, target_words, topic):
        """Format the response based on mode and target word count with IMPROVED word count targeting."""
        
        newline = "\n"
        
        if mode.lower() == "learning":
            response = f"""**Academic Explanation: {topic}**

**Definition:**
{content_dict['definition']}

**Key Characteristics:**
{newline.join(f"• {char}" for char in content_dict['characteristics'][:3])}

**Practical Examples:**
{newline.join(f"• {ex}" for ex in content_dict['examples'][:3])}

**Applications:**
{newline.join(f"• {app}" for app in content_dict['applications'][:3])}

**Conclusion:**
Understanding {topic.lower()} is essential for advancing knowledge in computer science and developing practical solutions that address real-world challenges. This field continues to evolve with new research and technological developments."""
        else:
            # IMPROVED: Better word count targeting for question mode
            if not target_words:
                target_words = 100
            
            if target_words <= 100:
                # For 2 marks (100 words) - keep it concise
                response = f"**Definition:** {content_dict['definition'][:120]}... **Key Characteristics:** {content_dict['characteristics'][0][:80]}... **Examples:** {content_dict['examples'][0][:60]}..."
                
                # Expand to reach closer to 100 words
                current_words = len(response.split())
                if current_words < 80:
                    response += f" {content_dict['characteristics'][1][:50]}..."
            
            elif target_words <= 250:
                # For 5 marks (250 words) - balanced detail
                response = f"**Definition:** {content_dict['definition']}"
                response += f"{newline}{newline}**Key Characteristics:** "
                response += f"{newline.join(f'• {char}' for char in content_dict['characteristics'][:2])}"
                response += f"{newline}{newline}**Examples:** "
                response += f"{newline.join(f'• {ex}' for ex in content_dict['examples'][:2])}"
                
                # Check word count and expand if needed
                current_words = len(response.split())
                if current_words < target_words * 0.8:  # If under 80% of target
                    response += f"{newline}{newline}**Applications:** "
                    response += f"{newline.join(f'• {app}' for app in content_dict['applications'][:2])}"
                    
                    # Add more content if still short
                    current_words = len(response.split())
                    if current_words < target_words * 0.9:
                        response += f" {content_dict['characteristics'][2]} This demonstrates the comprehensive nature of the field and its practical significance in modern technology applications."
            
            else:  # 500 words
                # For 10 marks (500 words) - comprehensive coverage
                response = f"""**Definition:** {content_dict['definition']}

**Key Characteristics:**
{newline.join(f"• {char}" for char in content_dict['characteristics'][:4])}

**Practical Examples:**
{newline.join(f"• {ex}" for ex in content_dict['examples'][:4])}

**Applications:**
{newline.join(f"• {app}" for app in content_dict['applications'][:3])}

**Significance:**
{topic} plays a crucial role in modern technology and continues to drive innovation across multiple industries. Understanding these concepts is essential for professionals in computer science and related fields."""
                
                # Expand further if needed for 500 words
                current_words = len(response.split())
                if current_words < target_words * 0.8:
                    response += f"{newline}{newline}**Advanced Concepts:** The field encompasses both theoretical foundations and practical implementations. Research continues to advance our understanding and develop new methodologies. Contemporary developments include optimization techniques, scalability improvements, and integration with emerging technologies. The interdisciplinary nature contributes to its broad applicability across various domains and industries."
                    
                    # Final expansion if still short
                    current_words = len(response.split())
                    if current_words < target_words * 0.9:
                        response += f" Future developments promise even greater capabilities and applications, making this an exciting area for continued study and professional development."
        
        return response

    def _enhance_and_expand_response(self, answer, query, mode, target_words):
        """Enhanced response expansion with better word count targeting."""
        
        if not answer.strip():
            return self._generate_expanded_response(query, mode, None, target_words, [])
        
        # Check if response needs enhancement
        current_words = len(answer.split())
        
        # IMPROVED: More aggressive expansion for question mode
        if mode.lower() == "question" and target_words:
            if current_words < target_words * 0.6:  # If significantly under target
                return self._generate_expanded_response(query, mode, None, target_words, [])
            elif current_words < target_words * 0.8:  # If moderately under target
                # Add targeted expansion
                expansion = self._get_targeted_expansion(query, target_words - current_words)
                answer += f" {expansion}"
        
        # Add academic structure if missing
        if not any(marker in answer.lower() for marker in ['definition:', '**definition', 'characteristics:', 'examples:']):
            if mode.lower() == "learning":
                structured = f"**Academic Explanation:**\n\n{answer}\n\nThis provides a comprehensive overview of the topic with practical implications for further study and application."
            else:
                structured = f"**Academic Answer:**\n{answer}"
            return structured
        
        return answer

    def _get_targeted_expansion(self, query, words_needed):
        """Generate targeted expansion text based on query topic."""
        query_lower = query.lower()
        
        if words_needed <= 0:
            return ""
        
        expansions = []
        
        if any(term in query_lower for term in ['algorithm', 'algorithms']):
            expansions = [
                "Algorithm design involves careful consideration of time and space complexity.",
                "Common algorithmic paradigms include divide-and-conquer, dynamic programming, and greedy approaches.",
                "The choice of algorithm significantly impacts program performance and scalability.",
                "Computational complexity analysis helps determine efficiency using Big O notation.",
                "Modern algorithm development focuses on optimization techniques and parallel processing capabilities."
            ]
        elif any(term in query_lower for term in ['machine learning', 'ml']):
            expansions = [
                "Machine learning algorithms are categorized into supervised, unsupervised, and reinforcement learning.",
                "Performance evaluation involves metrics such as accuracy, precision, recall, and F1-score.",
                "Contemporary applications span healthcare diagnostics, financial modeling, and autonomous systems.",
                "The field encompasses statistical methods, neural networks, and deep learning architectures.",
                "Training data quality and feature engineering are crucial for model effectiveness."
            ]
        elif any(term in query_lower for term in ['neural', 'network']):
            expansions = [
                "Neural network architectures vary significantly based on application requirements.",
                "Training involves forward propagation for prediction and backpropagation for weight adjustment.",
                "Activation functions such as ReLU, sigmoid, and tanh determine neuron processing.",
                "Contemporary developments include attention mechanisms and transformer architectures.",
                "Network topology affects learning capacity and computational requirements."
            ]
        else:
            expansions = [
                "This field encompasses both theoretical foundations and practical implementations.",
                "Research continues to advance understanding and develop new methodologies.",
                "The interdisciplinary nature contributes to broad applicability across domains.",
                "Contemporary developments continue to shape the field with new approaches.",
                "Understanding these concepts is essential for professional development."
            ]
        
        # Select expansions to meet word count
        result = ""
        current_words = 0
        
        for expansion in expansions:
            expansion_words = len(expansion.split())
            if current_words + expansion_words <= words_needed:
                result += f" {expansion}"
                current_words += expansion_words
            else:
                # Truncate last expansion if needed
                remaining_words = words_needed - current_words
                if remaining_words > 5:  # Only add if meaningful
                    words = expansion.split()[:remaining_words]
                    result += f" {' '.join(words)}"
                break
        
        return result.strip()

    def _expand_to_target_words(self, answer, query, target_words, mode):
        """IMPROVED: Expand the response to meet target word count with better accuracy."""
        
        current_words = len(answer.split())
        if current_words >= target_words * 0.9:  # Close enough (within 10%)
            return answer
        
        words_needed = target_words - current_words
        
        # Get targeted expansion
        expansion = self._get_targeted_expansion(query, words_needed)
        
        if expansion:
            expanded_answer = answer + " " + expansion
            
            # Fine-tune to get closer to target
            final_words = len(expanded_answer.split())
            if final_words > target_words * 1.1:  # If over target by more than 10%
                # Truncate to target
                words = expanded_answer.split()[:target_words]
                expanded_answer = ' '.join(words)
                # Try to end at sentence boundary
                if not expanded_answer.endswith('.'):
                    expanded_answer += '.'
            
            return expanded_answer
        
        return answer

    def _final_quality_check(self, answer, query, mode, target_words):
        """IMPROVED: Perform final quality checks with better word count enforcement."""
        
        # Remove any remaining artifacts
        answer = re.sub(r'\\\+', '*', answer)
        answer = re.sub(r'---+', '', answer)
        answer = re.sub(r'===+', '', answer)
        
        # Ensure proper capitalization
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Ensure proper ending
        if answer and not answer.rstrip().endswith(('.', '!', '?', ':')):
            answer = answer.rstrip() + '.'
        
        # IMPROVED: Stricter word count enforcement for question mode
        if mode.lower() == "question" and target_words:
            current_words = len(answer.split())
            variance = abs(current_words - target_words) / target_words
            
            if variance > 0.3:  # If variance is more than 30%
                if current_words < target_words:
                    # Expand more aggressively
                    words_needed = target_words - current_words
                    expansion = self._get_targeted_expansion(query, words_needed)
                    if expansion:
                        answer += f" {expansion}"
                else:
                    # Truncate more precisely
                    words = answer.split()[:target_words]
                    answer = ' '.join(words)
                    if not answer.endswith('.'):
                        answer += '.'
        
        # Minimum length check for learning mode
        if mode.lower() == "learning" and len(answer.split()) < 80:
            answer += "\n\nThis overview provides essential foundational knowledge for understanding the topic. Further exploration through academic literature and practical experience will deepen comprehension and enable advanced application of these concepts."
        
        return answer.strip()

    def _rerank_context(self, query, context_chunks):
        """Advanced reranker that filters out irrelevant content."""
        query_words = set(query.lower().split())
        
        # Terms that indicate irrelevant content
        irrelevant_terms = {'steve martin', 'millionaire', 'caveat lector', 'bibliography', 'acknowledgments', 'preface'}
        
        # Academic terms that boost relevance
        academic_terms = {'algorithm', 'machine learning', 'neural', 'computer', 'data', 'system', 'method', 'approach', 'theory', 'model'}
        
        def score_chunk(chunk):
            chunk_lower = chunk.lower()
            
            # Heavily penalize irrelevant content
            if any(term in chunk_lower for term in irrelevant_terms):
                return -1000
            
            # Base score from query word matches
            base_score = sum(word in chunk_lower for word in query_words) * 10
            
            # Bonus for academic terms
            academic_bonus = sum(3 for term in academic_terms if term in chunk_lower)
            
            # Bonus for technical content indicators
            technical_bonus = 0
            if any(word in chunk_lower for word in ['definition', 'example', 'characteristic', 'principle']):
                technical_bonus += 5
            
            # Penalty for very short or very long chunks
            length_factor = 1.0
            if len(chunk) < 50:
                length_factor = 0.5
            elif len(chunk) > 1000:
                length_factor = 0.8
            
            return (base_score + academic_bonus + technical_bonus) * length_factor
        
        ranked_chunks = sorted(context_chunks, key=score_chunk, reverse=True)
        # Only return chunks with positive scores
        return [chunk for chunk in ranked_chunks if score_chunk(chunk) > 0]

    def _write_logs(self, logs):
        """Append logs to session log file with user context."""
        try:
            session_log = f"quillai_llm_session_{datetime.utcnow().strftime('%Y%m%d')}.log"
            with open(session_log, "a", encoding="utf-8") as logf:
                logf.write(f"\n{'='*20} AbhayRao38 {datetime.utcnow().strftime('%H:%M:%S')} {'='*20}\n")
                logf.write("\n".join(logs))
                logf.write("\n" + "=" * 80 + "\n")
                logf.flush()  # FIXED: Added flush for real-time log inspection
        except Exception as e:
            print(f"Warning: Could not write logs: {e}")

    def get_model_info(self):
        """Return information about the loaded model."""
        return {
            "model_name": getattr(self.model.config, '_name_or_path', 'unknown'),
            "model_type": getattr(self.model.config, 'model_type', 'unknown'),
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": 1024,  # DialoGPT limit
            "device": self.device,
            "current_user": "AbhayRao38",
            "session_time": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + " UTC",
            "semantic_model_available": self.semantic_model is not None
        }

    def analyze_query_complexity(self, query):
        """Analyze the complexity and type of the user's query with enhanced detection."""
        query_lower = query.lower()
        
        complexity_indicators = {
            'simple': ['what is', 'define', 'meaning of'],
            'medium': ['explain', 'how does', 'describe', 'compare'],
            'complex': ['analyze', 'evaluate', 'synthesize', 'critically examine']
        }
        
        query_types = {
            'definition': ['define', 'definition', 'what is', 'what are', 'meaning'],
            'explanation': ['explain', 'describe', 'how', 'why', 'process'],
            'comparison': ['compare', 'contrast', 'difference', 'versus', 'vs'],
            'analysis': ['analyze', 'examine', 'evaluate', 'assess'],
            'application': ['apply', 'use', 'implement', 'example'],
            'question_generation': ['generate', 'create', 'make', 'questions'],
            'summary': ['summarize', 'summary', 'key points', 'overview']
        }
        
        # Determine complexity
        complexity = 'simple'
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity = level
                break
        
        # Determine type
        query_type = 'general'
        for qtype, indicators in query_types.items():
            if any(indicator in query_lower for indicator in indicators):
                query_type = qtype
                break
        
        # Detect domain
        domain, topics, confidence = self.detect_domain_and_topic(query)
        
        return {
            'complexity': complexity,
            'type': query_type,
            'domain': domain,
            'topics': topics,
            'length': len(query.split()),
            'academic_terms': sum(1 for term in ['theory', 'concept', 'principle', 'algorithm', 'method'] if term in query_lower)
        }

    def _extract_topic_from_query(self, query):
        """Extract meaningful topic from query with enhanced processing."""
        # Remove common question words and phrases
        topic = re.sub(r'\b(what|is|are|how|why|when|where|explain|define|describe|tell|me|about|the|a|an|generate|create|make|questions?|for|of)\b', '', query.lower(), flags=re.IGNORECASE)
        topic = re.sub(r'[^\w\s]', '', topic).strip()
        topic = re.sub(r'\s+', ' ', topic)
        
        # Remove numbers (often from question generation requests)
        topic = re.sub(r'\b\d+\b', '', topic).strip()
        
        # Take meaningful words
        words = [w for w in topic.split() if len(w) > 2][:4]
        
        if words:
            return ' '.join(words)
        else:
            return "Academic Topic"

    def _infer_target_word_count(self, marks):
        """Infer target word count from marks."""
        if marks is None:
            return None
        return {2: 100, 5: 250, 10: 500}.get(marks, 100)

    def generate_dual_response(self, query, mode="learning", marks=None, temperature=0.7, context_chunks=None):
        """
        IMPROVED: Generate both LLM and enhanced custom academic responses independently.
        
        Args:
            query: The user's question/prompt
            mode: "learning" for detailed responses, "question" for concise answers
            marks: For question mode, affects target word count (2/5/10 -> 100/250/500 words)
            temperature: Sampling temperature for LLM generation
            context_chunks: List of relevant text chunks to include as context
            
        Returns:
            Dict with both outputs and metadata
        """
        logs = []
        start_time = datetime.utcnow()
        context_chunks = context_chunks or []
        target_words = self._infer_target_word_count(marks)
        
        logs.append(f"=== ENHANCED DUAL RESPONSE GENERATION ===")
        logs.append(f"Query: {query}")
        logs.append(f"Mode: {mode}, Marks: {marks}, Target words: {target_words}")
        logs.append(f"Temperature: {temperature}")
        logs.append(f"Context chunks: {len(context_chunks)}")
        
        # 1. Generate from LLM (DialoGPT-medium)
        llm_start_time = datetime.utcnow()
        logs.append(f"--- Starting LLM Generation ---")
        
        try:
            llm_answer = self.generate_answer(
                query=query,
                mode=mode,
                marks=marks,
                temperature=temperature,
                context_chunks=context_chunks,
                rerank_context=True,
                return_citations=True
            )
            llm_generation_time = (datetime.utcnow() - llm_start_time).total_seconds()
            llm_word_count = len(llm_answer.split())
            logs.append(f"LLM generation successful: {llm_word_count} words in {llm_generation_time:.2f}s")
            
        except Exception as e:
            logs.append(f"LLM generation failed: {repr(e)}")
            # Use fallback for LLM
            llm_answer = self._generate_expanded_response(query, mode, marks, target_words, context_chunks)
            llm_generation_time = (datetime.utcnow() - llm_start_time).total_seconds()
            llm_word_count = len(llm_answer.split())
            logs.append(f"LLM fallback used: {llm_word_count} words in {llm_generation_time:.2f}s")
        
        # 2. IMPROVED: Generate from enhanced custom academic generator
        custom_start_time = datetime.utcnow()
        logs.append(f"--- Starting Enhanced Custom Academic Generation ---")
        
        try:
            # Enhanced keyword detection and topic-specific generation
            custom_answer = self._generate_enhanced_custom_response(query, mode, marks, target_words, context_chunks)
            
            custom_generation_time = (datetime.utcnow() - custom_start_time).total_seconds()
            custom_word_count = len(custom_answer.split())
            logs.append(f"Enhanced custom generation successful: {custom_word_count} words in {custom_generation_time:.2f}s")
            
        except Exception as e:
            logs.append(f"Custom generation failed: {repr(e)}")
            # Fallback for custom generator
            custom_answer = self._generate_expanded_response(query, mode, marks, target_words, context_chunks)
            custom_generation_time = (datetime.utcnow() - custom_start_time).total_seconds()
            custom_word_count = len(custom_answer.split())
            logs.append(f"Custom fallback used: {custom_word_count} words in {custom_generation_time:.2f}s")
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        logs.append(f"=== ENHANCED DUAL GENERATION COMPLETE ===")
        logs.append(f"Total time: {total_time:.2f}s")
        logs.append(f"LLM: {llm_word_count} words, Enhanced Custom: {custom_word_count} words")
        
        # Write logs
        self._write_logs(logs)
        
        # Return structured response
        result = {
            "llm_output": llm_answer,
            "custom_output": custom_answer,
            "word_counts": {
                "llm": llm_word_count,
                "custom": custom_word_count
            },
            "generation_times": {
                "llm": llm_generation_time,
                "custom": custom_generation_time,
                "total": total_time
            },
            "metadata": {
                "query": query,
                "mode": mode,
                "marks": marks,
                "target_words": target_words,
                "temperature": temperature,
                "context_chunks_used": len(context_chunks),
                "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + " UTC"
            }
        }
        
        return result

    def _generate_enhanced_custom_response(self, query, mode, marks, target_words, context_chunks):
        """
        ENHANCED: Generate custom academic response with improved word count targeting and query specificity.
        """
        query_lower = query.lower()
        
        # Enhanced topic detection with more specific matching
        topic_detected = None
        specific_subtopic = None
        
        # More granular topic detection
        if any(term in query_lower for term in ['divide and conquer', 'divide-and-conquer']):
            topic_detected = 'divide_conquer'
            specific_subtopic = 'divide and conquer'
        elif any(term in query_lower for term in ['time complexity', 'space complexity', 'complexity']):
            topic_detected = 'complexity'
            specific_subtopic = 'algorithmic complexity'
        elif any(term in query_lower for term in ['greedy algorithm', 'greedy']):
            topic_detected = 'greedy'
            specific_subtopic = 'greedy algorithms'
        elif any(term in query_lower for term in ['dynamic programming', 'dp']):
            topic_detected = 'dynamic_programming'
            specific_subtopic = 'dynamic programming'
        elif any(term in query_lower for term in ['sorting algorithm', 'sorting']):
            topic_detected = 'sorting'
            specific_subtopic = 'sorting algorithms'
        elif any(term in query_lower for term in ['search algorithm', 'searching']):
            topic_detected = 'searching'
            specific_subtopic = 'search algorithms'
        elif any(term in query_lower for term in ['machine learning', 'ml']):
            topic_detected = 'machine_learning'
            specific_subtopic = 'machine learning'
        elif any(term in query_lower for term in ['neural network', 'neural']):
            topic_detected = 'neural_networks'
            specific_subtopic = 'neural networks'
        elif any(term in query_lower for term in ['artificial intelligence', 'ai']):
            topic_detected = 'artificial_intelligence'
            specific_subtopic = 'artificial intelligence'
        elif any(term in query_lower for term in ['algorithm', 'algorithms']):
            topic_detected = 'algorithms'
            specific_subtopic = 'algorithms'
        else:
            topic_detected = 'generic'
            specific_subtopic = self._extract_topic_from_query(query)
        
        # Generate topic-specific response with precise word targeting
        if topic_detected == 'divide_conquer':
            response = self._generate_divide_conquer_response(mode, target_words)
        elif topic_detected == 'complexity':
            response = self._generate_complexity_response(mode, target_words)
        elif topic_detected == 'greedy':
            response = self._generate_greedy_response(mode, target_words)
        elif topic_detected == 'dynamic_programming':
            response = self._generate_dp_response(mode, target_words)
        elif topic_detected == 'sorting':
            response = self._generate_sorting_response(mode, target_words)
        elif topic_detected == 'searching':
            response = self._generate_searching_response(mode, target_words)
        elif topic_detected == 'machine_learning':
            response = self._generate_ml_response(mode, target_words)
        elif topic_detected == 'neural_networks':
            response = self._generate_neural_response(mode, target_words)
        elif topic_detected == 'artificial_intelligence':
            response = self._generate_ai_response(mode, target_words)
        elif topic_detected == 'algorithms':
            response = self._generate_algorithm_response(mode, target_words)
        else:
            response = self._generate_enhanced_generic_response(query, mode, target_words, specific_subtopic)
        
        # CRITICAL: Apply precise word count targeting
        response = self._apply_precise_word_targeting(response, target_words, mode, query)
        
        return response

    def _apply_precise_word_targeting(self, response, target_words, mode, query):
        """Apply precise word count targeting to meet exact requirements."""
        
        if not target_words:
            return response
        
        current_words = len(response.split())
        variance = abs(current_words - target_words) / target_words
        
        # If variance is acceptable (within 15%), return as is
        if variance <= 0.15:
            return response
        
        # If significantly under target, expand
        if current_words < target_words * 0.8:
            words_needed = target_words - current_words
            expansion = self._get_targeted_expansion(query, words_needed)
            if expansion:
                response += f" {expansion}"
        
        # If over target, truncate precisely
        elif current_words > target_words * 1.1:
            words = response.split()[:target_words]
            response = ' '.join(words)
            # Ensure proper ending
            if not response.endswith('.'):
                response += '.'
        
        # Final fine-tuning to get as close as possible to target
        final_words = len(response.split())
        if abs(final_words - target_words) > target_words * 0.1:  # If still off by more than 10%
            if final_words < target_words:
                # Add filler content to reach target
                remaining = target_words - final_words
                if remaining > 5:
                    filler = "This comprehensive understanding enables effective application in professional and academic contexts."
                    filler_words = filler.split()[:remaining]
                    response += f" {' '.join(filler_words)}"
            else:
                # Precise truncation
                words = response.split()[:target_words]
                response = ' '.join(words)
                if not response.endswith('.'):
                    response += '.'
        
        return response

# === Test and demonstration ===
if __name__ == "__main__":
    print("="*60)
    print("Initializing Enhanced QuillAI LLM with Advanced Features...")
    print(f"User: AbhayRao38")
    print(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*60)
    
    try:
        # Initialize model
        model = QuillAILLM(model_name="microsoft/DialoGPT-medium", force_model_check=True)
        
        # Test enhanced features
        test_cases = [
            ("Generate 5 questions about machine learning", "learning", None),
            ("Create a rubric for algorithms assessment", "question", 5),
            ("What is artificial intelligence?", "question", 5),
            ("Compare supervised and unsupervised learning", "question", 10),
            ("List 3 examples of sorting algorithms", "learning", None)
        ]
        
        for query, mode, marks in test_cases:
            print(f"\n--- Enhanced Test: {query} (Mode: {mode}, Marks: {marks}) ---")
            try:
                # Test intent detection
                intent, confidence, all_intents = model.detect_query_intent(query)
                print(f"Intent: {intent} (confidence: {confidence:.2f})")
                
                # Test domain detection
                domain, topics, domain_confidence = model.detect_domain_and_topic(query)
                print(f"Domain: {domain}, Topics: {topics}")
                
                answer = model.generate_answer(query, mode=mode, marks=marks)
                word_count = len(answer.split())
                print(f"✓ Success: {word_count} words")
                print(f"Preview: {answer[:200]}...")
            except Exception as e:
                print(f"✗ Failed: {e}")
    
    except Exception as ex:
        print(f"FAIL: {ex}")
        import traceback
        traceback.print_exc()