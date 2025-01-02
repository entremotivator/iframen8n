import streamlit as st
import requests

# Streamlit App Title
st.title("Ollama API Query Interface")

# Instructions
st.markdown("### Interact with the Ollama API via this interface. Enter your query below:")

# User Input
query = st.text_area("Enter your query:", placeholder="Type something to query Ollama...")

# Submit Button
if st.button("Submit"):
    if query.strip():
        # Send Query to Ollama API
        try:
            response = requests.post(
                "https://theaisource-u29564.vm.elestio.app:57987/query",
                auth=("root", "eZfLK3X4-SX0i-UmgUBe6E"),
                json={"prompt": query},
            )
            if response.status_code == 200:
                # Display API Response
                result = response.json()
                st.success("Query Successful!")
                st.write("**Response:**")
                st.json(result)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query before submitting.")

# Footer Information
st.markdown("---")
st.markdown("🔒 **Note:** API credentials are securely used within this app and not exposed.")
