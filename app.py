
import streamlit as st

from src.rag_pipeline import answer_question


st.set_page_config(
    page_title="NovaMart Support Assistant",
    page_icon="🛍️",
    layout="centered"
)

st.title("NovaMart Support Assistant")
st.caption(
    "A retrieval-augmented customer support assistant that answers "
    "questions using NovaMart's policy documents."
)

with st.sidebar:
    st.header("About this project")

    st.markdown(
        """
        This assistant uses:

        - Local sentence-transformer embeddings
        - ChromaDB semantic retrieval
        - OpenAI answer generation
        - Source attribution
        - Grounded refusal for unsupported questions
        """
    )

    st.header("Example questions")

    st.markdown(
        """
        - Can I return an unused item after 30 days?
        - My order arrived damaged. What should I do?
        - How long is the product warranty?
        - Can I cancel an order after it ships?
        - How long is a password reset link valid?
        """
    )

with st.form("customer_question_form"):
    question = st.text_area(
        "Ask a customer-support question",
        placeholder=(
            "Example: My order arrived damaged. What should I do?"
        ),
        height=120
    )

    submitted = st.form_submit_button(
        "Ask NovaMart",
        use_container_width=True
    )

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching NovaMart policies..."):
            try:
                result = answer_question(question.strip())

            except Exception as error:
                st.error(
                    "The assistant could not generate an answer. "
                    "Please check the application setup and try again."
                )

                with st.expander("Technical details"):
                    st.code(str(error))

            else:
                st.subheader("Answer")
                st.write(result["answer"])

                refusal_text = (
                    "I cannot answer this from the available "
                    "NovaMart policies."
                )

                if result["answer"].strip() == refusal_text:
                    st.info(
                        "No directly supported NovaMart policy was found."
                    )

                else:
                    st.subheader("Retrieved evidence")

                    st.caption(
                        "These are the policy sections retrieved before "
                        "the answer was generated."
                    )

                    for position, chunk in enumerate(
                        result["retrieved_chunks"],
                        start=1
                    ):
                        label = (
                            f"{position}. {chunk['source']} "
                            f"— similarity {chunk['similarity']:.3f}"
                        )

                        with st.expander(label):
                            st.markdown(chunk["content"])
