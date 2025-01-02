import streamlit as st

# App Title
st.title("Embedded Agent Online")

# App Description
st.write(
    "This is a Streamlit app embedding the Agent Online webpage within an iframe."
)

# Embed the iframe
iframe_url = "https://agentonline-u29564.vm.elestio.app"
st.components.v1.iframe(iframe_url, width=800, height=600)

# Footer
st.write("Powered by Streamlit")
