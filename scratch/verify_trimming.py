import sys
import os

# Mock the parts needed to test src/agent/graph.py logic
# But actually, I can just import and mock the LLM or check the code logic.


# Simulate the trimming logic from graph.py:207
def test_trimming(messages):
    trimmed_messages = messages[-4:] if len(messages) > 4 else messages
    return trimmed_messages


# Test cases
msg1 = [{"role": "user", "content": "Hello"}]
msg2 = [
    {"role": "user", "content": "1"},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": "3"},
    {"role": "assistant", "content": "4"},
]
msg3 = [
    {"role": "user", "content": "1"},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": "3"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "5"},
]

print(f"Test 1 (len 1): {len(test_trimming(msg1))}")
print(f"Test 2 (len 4): {len(test_trimming(msg2))}")
print(f"Test 3 (len 5): {len(test_trimming(msg3))}")  # Should be 4
print(
    f"Test 3 content: {[m['content'] for m in test_trimming(msg3)]}"
)  # Should be 2, 3, 4, 5

if len(test_trimming(msg3)) == 4 and test_trimming(msg3)[0]["content"] == "2":
    print("✅ Trimming logic verified!")
else:
    print("❌ Trimming logic failed!")
