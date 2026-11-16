"""Demo inputs for the Streamlit app (written by us for the demo, NOT training data)."""

SINGLE_SAMPLES = {
    "Strong answer": (
        "Explain how database indexing improves query performance.",
        "An index is a separate data structure, usually a B-tree, that lets the database find rows "
        "without scanning the whole table. I added a composite index on the user_id and created_at "
        "columns of our orders table and reduced the query time from 2 seconds to 40 milliseconds. "
        "The trade-off is slower writes and extra storage, so I only index columns used in frequent filters.",
    ),
    "Hesitant answer": (
        "Explain how database indexing improves query performance.",
        "Um, so, like, I think an index is kind of a thing that makes the database faster, maybe? "
        "I'm not really sure how it works inside, I just, you know, asked my teammate to add it "
        "and I hoped it would help.",
    ),
    "Off-topic answer": (
        "Explain how database indexing improves query performance.",
        "I really enjoy working in teams and I am a quick learner. Last weekend I travelled with my "
        "friends and we organised the whole trip ourselves, which taught me a lot about planning "
        "and communication.",
    ),
}

FULL_TRANSCRIPT_SAMPLE = """Interviewer: Can you explain the difference between processes and threads?
Candidate: A process has its own memory space, while threads inside one process share memory. Because of that, context switching between threads is cheaper, but I have to protect shared data with locks. In my last project I used a thread pool to handle 500 concurrent requests and reduced response time by 30 percent.
Interviewer: How would you design a URL shortening service?
Candidate: Um, I think, like, you would store the long URL in a database and, you know, generate some kind of short id, maybe with hashing? I'm not totally sure about the scaling part.
Interviewer: Tell me how you handle a disagreement with a teammate.
Candidate: I set up a short meeting, listened to their reasoning, and proposed a quick benchmark so we could decide using data. We implemented the faster option and documented the decision for the team."""
