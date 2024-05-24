import streamlit as st

# Define the exercises
exercises = {
    11: {
        "title": "Subtraction",
        "questions": [
            {"question": "5 - 2 = ", "answer": 3},
            {"question": "7 - 8 = ", "answer": -1},
            {"question": "6 - .. = 3", "answer": 3},
            {"question": ".. - 4 = 0", "answer": 4}
        ]
    },
    12: {
        "title": "Missing Word",
        "questions": [
            {"question": "The girl was playing with .... doll", "answer": "her"}
        ]
    },
    13: {
        "title": "Multiplication",
        "questions": [
            {"question": "3 * 4 = ", "answer": 12},
            {"question": "5 * 6 = ", "answer": 30},
            {"question": "7 * 8 = ", "answer": 56},
            {"question": "9 * 9 = ", "answer": 81}
        ]
    }
}

# Function to evaluate responses
def evaluate_responses(exercise_id, responses):
    correct_answers = [q['answer'] for q in exercises[exercise_id]['questions']]
    correct_count = sum(1 for correct, response in zip(correct_answers, responses) if correct == response)
    return correct_count, len(correct_answers)

# Streamlit app
st.title("Interactive Exercise System")

# Sidebar for exercise selection
exercise_options = {v['title']: k for k, v in exercises.items()}
selected_exercise_title = st.sidebar.selectbox("Select an Exercise", list(exercise_options.keys()))
selected_exercise_id = exercise_options[selected_exercise_title]

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}

# Display the current exercise
st.header(f"Exercise {selected_exercise_id}: {exercises[selected_exercise_id]['title']}")

responses = []
for i, question in enumerate(exercises[selected_exercise_id]['questions']):
    response = st.text_input(question['question'], key=f"q{selected_exercise_id}_{i}")
    responses.append(response)

if st.button("Submit"):
    # Convert responses to correct types (int/str)
    correct_responses = []
    for response in responses:
        try:
            correct_responses.append(int(response))
        except ValueError:
            correct_responses.append(response.strip())

    correct, total = evaluate_responses(selected_exercise_id, correct_responses)
    st.session_state.results[selected_exercise_id] = {"correct": correct, "total": total}

    # Feedback and progress to next task if applicable
    if correct == total:
        st.success(f"Congratulations! You have successfully completed exercise {selected_exercise_id}.")
    else:
        st.warning(f"You need more practice on exercise {selected_exercise_id}.")

# Update student profile
def update_student_profile(results):
    student_profile = {
        "fully_understood": [],
        "somewhat_understood": [],
        "needs_improvement": []
    }

    for exercise_id, result in results.items():
        if result['correct'] == result['total']:
            student_profile["fully_understood"].append(exercise_id)
        elif result['correct'] >= (0.5 * result['total']):
            student_profile["somewhat_understood"].append(exercise_id)
        else:
            student_profile["needs_improvement"].append(exercise_id)

    return student_profile

if st.button("Show Profile"):
    student_profile = update_student_profile(st.session_state.results)
    st.write(f"Student Profile: {student_profile}")
