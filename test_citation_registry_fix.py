"""
Test to verify CitationRegistry race condition fix.

This test demonstrates that the CitationRegistry now uses thread-local storage
to prevent cross-request contamination in concurrent scenarios.

Bug: https://github.com/OpenBMB/UltraRAG/issues/394
"""

import threading
import sys
sys.path.insert(0, '/tmp/ultrarag-worktree/servers/custom/src')

from custom import CitationRegistry, SurveyCPMCitationRegistry


def test_citation_registry_thread_isolation():
    """Test that CitationRegistry isolates state between threads."""
    results = {}
    
    def worker(thread_id):
        """Simulate a request that initializes and uses citation registry."""
        # Each thread resets and uses the registry
        CitationRegistry.reset()
        
        # Assign some citations
        id1 = CitationRegistry.assign_id(0, "document 1")
        id2 = CitationRegistry.assign_id(0, "document 2")
        id3 = CitationRegistry.assign_id(0, "document 1")  # Should return same as id1
        
        results[thread_id] = {
            'id1': id1,
            'id2': id2,
            'id3': id3,
        }
    
    # Run multiple threads concurrently
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Verify each thread got consistent results
    print("=== Thread Results ===")
    for thread_id, result in sorted(results.items()):
        print(f"Thread {thread_id}: id1={result['id1']}, id2={result['id2']}, id3={result['id3']}")
        assert result['id1'] == 1, f"Thread {thread_id}: First doc should be ID 1"
        assert result['id2'] == 2, f"Thread {thread_id}: Second doc should be ID 2"
        assert result['id3'] == 1, f"Thread {thread_id}: Duplicate doc should return ID 1"
    
    print("\n✅ All threads got consistent, isolated results!")
    print("✅ No cross-request contamination detected!")
    return True


def test_surveycpm_registry_thread_isolation():
    """Test that SurveyCPMCitationRegistry isolates state between threads."""
    results = {}
    
    def worker(thread_id):
        """Simulate a request that initializes and uses SurveyCPM registry."""
        SurveyCPMCitationRegistry.reset()
        
        id1 = SurveyCPMCitationRegistry.assign_id(0, "survey doc 1")
        id2 = SurveyCPMCitationRegistry.assign_id(0, "survey doc 2")
        
        results[thread_id] = {
            'id1': id1,
            'id2': id2,
        }
    
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("\n=== SurveyCPM Thread Results ===")
    for thread_id, result in sorted(results.items()):
        print(f"Thread {thread_id}: id1={result['id1']}, id2={result['id2']}")
        assert result['id1'] == 'textid1', f"Thread {thread_id}: First doc should be textid1"
        assert result['id2'] == 'textid2', f"Thread {thread_id}: Second doc should be textid2"
    
    print("\n✅ SurveyCPM registry also thread-safe!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CitationRegistry thread isolation fix")
    print("Bug: https://github.com/OpenBMB/UltraRAG/issues/394")
    print("=" * 60)
    
    test_citation_registry_thread_isolation()
    test_surveycpm_registry_thread_isolation()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
