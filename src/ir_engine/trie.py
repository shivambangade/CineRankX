"""Trie for prefix-based autocomplete over movie titles."""


class TrieNode:
    """Single node in a Trie."""

    def __init__(self):
        self.children = {}
        self.movies = []  # List of (movie_id, title) tuples that end at or pass through this node


class Trie:
    """Prefix tree for autocomplete searches over movie titles."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, title: str, movie_id: int) -> None:
        """Insert a movie title into the Trie."""
        if not isinstance(title, str) or not title.strip():
            return
        title_lower = title.lower().strip()
        node = self.root
        for char in title_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.movies.append((movie_id, title))

    def search(self, prefix: str, limit: int = 10) -> list[tuple[int, str]]:
        """Return movies whose titles start with prefix (case-insensitive)."""
        if not isinstance(prefix, str) or not prefix.strip():
            return []
        prefix_lower = prefix.lower().strip()
        node = self.root
        for char in prefix_lower:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []

        def collect_movies(n: TrieNode) -> None:
            if len(results) >= limit:
                return
            results.extend(n.movies[:limit - len(results)])
            for child in n.children.values():
                if len(results) >= limit:
                    break
                collect_movies(child)

        collect_movies(node)
        return results
