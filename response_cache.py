import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class DualResponseCache:
    """
    A simple file-based cache for dual LLM responses with expiry.
    Caches both LLM and custom outputs together.
    """
    def __init__(self, cache_dir: str = "/tmp/dual_response_cache", expiry_hours: int = 24):
        self.cache_dir = cache_dir
        self.expiry_seconds = expiry_hours * 3600
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"âœ… Initialized DualResponseCache at {self.cache_dir} with {expiry_hours}-hour expiry.")
        # Perform initial cleanup on startup
        self._cleanup_expired()

    def _get_cache_key(self, query: str, mode: str, marks: Optional[int]) -> str:
        """Generates a unique cache key based on query, mode, and marks."""
        # Normalize query for consistent caching
        normalized_query = query.strip().lower().replace(" ", "_")
        key_parts = [normalized_query, mode]
        if marks is not None:
            key_parts.append(str(marks))
        return "_".join(key_parts) + ".json"

    def get(self, query: str, mode: str, marks: Optional[int]) -> Optional[Dict[str, Any]]:
        """
        Retrieves a cached response if available and not expired.
        Verifies word counts on retrieval.
        """
        cache_key = self._get_cache_key(query, mode, marks)
        file_path = os.path.join(self.cache_dir, cache_key)

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                cache_timestamp = datetime.fromisoformat(cached_data['timestamp'].replace('Z', '+00:00'))
                if datetime.utcnow() - cache_timestamp < timedelta(seconds=self.expiry_seconds):
                    # Verify word counts for question mode
                    if mode == "question" and marks is not None:
                        target_words = {2: 100, 5: 250, 10: 500}.get(marks)
                        llm_output_wc = len(cached_data.get('llm_output', '').split())
                        custom_output_wc = len(cached_data.get('custom_output', '').split())

                        # Allow a small tolerance for cached word counts
                        if target_words and (
                            abs(llm_output_wc - min(target_words, 200)) / min(target_words, 200) > 0.1 or
                            abs(custom_output_wc - target_words) / target_words > 0.1
                        ):
                            print(f"Cache miss for {cache_key}: Word count mismatch. Recalculating.")
                            self.delete(query, mode, marks) # Invalidate cache
                            return None

                    print(f"Cache hit for {cache_key}")
                    return cached_data['response']
                else:
                    print(f"Cache miss for {cache_key}: Expired. Deleting.")
                    self.delete(query, mode, marks)
            except Exception as e:
                print(f"Error reading cache file {file_path}: {e}. Deleting corrupted entry.")
                self.delete(query, mode, marks)
        return None

    def set(self, query: str, mode: str, marks: Optional[int], response: Dict[str, Any]):
        """Stores a response in the cache."""
        cache_key = self._get_cache_key(query, mode, marks)
        file_path = os.path.join(self.cache_dir, cache_key)
        try:
            data_to_cache = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'query': query,
                'mode': mode,
                'marks': marks,
                'response': response,
                'llm_output': response.get('llm_output', ''), # Store for word count verification
                'custom_output': response.get('custom_output', '') # Store for word count verification
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_cache, f, indent=2, ensure_ascii=False)
            print(f"Cached response for {cache_key}")
        except Exception as e:
            print(f"Error writing cache file {file_path}: {e}")

    def delete(self, query: str, mode: str, marks: Optional[int]):
        """Deletes a specific cache entry."""
        cache_key = self._get_cache_key(query, mode, marks)
        file_path = os.path.join(self.cache_dir, cache_key)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted cache entry: {cache_key}")
            except Exception as e:
                print(f"Error deleting cache file {file_path}: {e}")

    def _cleanup_expired(self):
        """Removes expired cache entries."""
        now = datetime.utcnow()
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    cache_timestamp = datetime.fromisoformat(cached_data['timestamp'].replace('Z', '+00:00'))
                    if now - cache_timestamp > timedelta(seconds=self.expiry_seconds):
                        os.remove(file_path)
                        print(f"Cleaned up expired cache entry: {filename}")
                except Exception as e:
                    print(f"Error during cache cleanup for {filename}: {e}. Deleting corrupted entry.")
                    try:
                        os.remove(file_path)
                    except OSError as oe:
                        print(f"Failed to remove corrupted file {file_path}: {oe}")