import os
import requests
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import Ollama
from datetime import datetime
import io
import json

def run():
    st.title("Media Page")
    st.write("Welcome to the Media page!")
    st.write("Here you can manage media-related tasks and information.")

# API Configuration
DEFAULT_API_URL = "https://theaisource-u29564.vm.elestio.app:57987"
DEFAULT_USERNAME = "root"
DEFAULT_PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"

# Social Media Post Types and Platform-Specific Requirements
POST_TYPES = {
    "product_launch": {
        "name": "Product Launch",
        "fields": [
            "product_name",
            "key_features",
            "unique_selling_points",
            "price_point",
            "launch_date",
            "target_audience",
            "call_to_action"
        ]
    },
    "event_promotion": {
        "name": "Event Promotion",
        "fields": [
            "event_name",
            "date_time",
            "location",
            "description",
            "highlights",
            "registration_info",
            "early_bird_details"
        ]
    },
    "content_promotion": {
        "name": "Content Promotion",
        "fields": [
            "content_title",
            "content_type",
            "key_takeaways",
            "value_proposition",
            "content_link",
            "content_preview"
        ]
    },
    "company_update": {
        "name": "Company Update",
        "fields": [
            "update_type",
            "announcement",
            "impact",
            "future_implications",
            "stakeholder_benefits"
        ]
    },
    "customer_success": {
        "name": "Customer Success Story",
        "fields": [
            "customer_name",
            "industry",
            "challenge",
            "solution",
            "results",
            "testimonial"
        ]
    },
    "promotional_offer": {
        "name": "Promotional Offer",
        "fields": [
            "offer_details",
            "discount_amount",
            "validity_period",
            "terms_conditions",
            "redemption_process"
        ]
    }
}

PLATFORMS = {
    "linkedin": {
        "name": "LinkedIn",
        "max_length": 3000,
        "hashtag_limit": 5,
        "best_practices": [
            "Professional tone",
            "Industry insights",
            "Business focus",
            "Paragraph breaks",
            "Relevant hashtags"
        ]
    },
    "twitter": {
        "name": "Twitter",
        "max_length": 280,
        "hashtag_limit": 3,
        "best_practices": [
            "Concise messaging",
            "Engaging hooks",
            "Relevant hashtags",
            "Clear CTAs",
            "Rich media integration"
        ]
    },
    "instagram": {
        "name": "Instagram",
        "max_length": 2200,
        "hashtag_limit": 30,
        "best_practices": [
            "Visual focus",
            "Storytelling",
            "Emoji usage",
            "Hashtag grouping",
            "Engagement questions"
        ]
    },
    "facebook": {
        "name": "Facebook",
        "max_length": 63206,
        "hashtag_limit": 4,
        "best_practices": [
            "Conversational tone",
            "Rich media",
            "Question posts",
            "Community focus",
            "Event integration"
        ]
    }
}

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

def get_ollama_models():
    try:
        response = requests.get(
            f"{DEFAULT_API_URL}/api/tags",
            auth=(DEFAULT_USERNAME, DEFAULT_PASSWORD)
        )
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']
                    if all(keyword not in model['name'].lower()
                           for keyword in ('failed', 'embed', 'bge'))]
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def get_post_prompt(post_type: str, platform: str, form_data: dict) -> str:
    """Generate platform-specific prompts based on post type"""
    
    platform_info = PLATFORMS[platform]
    base_prompt = f"""Generate an optimized {platform_info['name']} post for a {POST_TYPES[post_type]['name']}.
    
    Platform Requirements:
    - Maximum length: {platform_info['max_length']} characters
    - Hashtag limit: {platform_info['hashtag_limit']}
    - Best practices: {', '.join(platform_info['best_practices'])}
    
    Content Details:
    """
    
    content_details = {
        "product_launch": f"""
    Product: {form_data['product_name']}
    Features: {form_data['key_features']}
    USP: {form_data['unique_selling_points']}
    Price: {form_data['price_point']}
    Launch Date: {form_data['launch_date']}
    Target Audience: {form_data['target_audience']}
    CTA: {form_data['call_to_action']}
    """,
        "event_promotion": f"""
    Event: {form_data['event_name']}
    Date/Time: {form_data['date_time']}
    Location: {form_data['location']}
    Description: {form_data['description']}
    Highlights: {form_data['highlights']}
    Registration: {form_data['registration_info']}
    Early Bird: {form_data['early_bird_details']}
    """,
        "content_promotion": f"""
    Title: {form_data['content_title']}
    Type: {form_data['content_type']}
    Key Takeaways: {form_data['key_takeaways']}
    Value Proposition: {form_data['value_proposition']}
    Link: {form_data['content_link']}
    Preview: {form_data['content_preview']}
    """,
        "company_update": f"""
    Update Type: {form_data['update_type']}
    Announcement: {form_data['announcement']}
    Impact: {form_data['impact']}
    Future Implications: {form_data['future_implications']}
    Benefits: {form_data['stakeholder_benefits']}
    """,
        "customer_success": f"""
    Customer: {form_data['customer_name']}
    Industry: {form_data['industry']}
    Challenge: {form_data['challenge']}
    Solution: {form_data['solution']}
    Results: {form_data['results']}
    Testimonial: {form_data['testimonial']}
    """,
        "promotional_offer": f"""
    Offer: {form_data['offer_details']}
    Discount: {form_data['discount_amount']}
    Validity: {form_data['validity_period']}
    Terms: {form_data['terms_conditions']}
    Redemption: {form_data['redemption_process']}
    """
    }
    
    return base_prompt + content_details[post_type]

def render_conditional_form(post_type):
    """Render form fields based on post type"""
    form_data = {}
    
    if post_type == "product_launch":
        form_data.update({
            'product_name': st.text_input("Product Name"),
            'key_features': st.text_area("Key Features"),
            'unique_selling_points': st.text_area("Unique Selling Points"),
            'price_point': st.text_input("Price Point"),
            'launch_date': st.date_input("Launch Date"),
            'target_audience': st.text_area("Target Audience"),
            'call_to_action': st.text_input("Call to Action")
        })
    
    elif post_type == "event_promotion":
        form_data.update({
            'event_name': st.text_input("Event Name"),
            'date_time': st.text_input("Date and Time"),
            'location': st.text_input("Location"),
            'description': st.text_area("Event Description"),
            'highlights': st.text_area("Event Highlights"),
            'registration_info': st.text_area("Registration Information"),
            'early_bird_details': st.text_area("Early Bird Details")
        })
    
    elif post_type == "content_promotion":
        form_data.update({
            'content_title': st.text_input("Content Title"),
            'content_type': st.selectbox("Content Type", ["Blog Post", "Video", "Podcast", "Whitepaper", "Webinar"]),
            'key_takeaways': st.text_area("Key Takeaways"),
            'value_proposition': st.text_area("Value Proposition"),
            'content_link': st.text_input("Content Link"),
            'content_preview': st.text_area("Content Preview")
        })
    
    elif post_type == "company_update":
        form_data.update({
            'update_type': st.selectbox("Update Type", ["Milestone", "New Partnership", "Achievement", "Company News"]),
            'announcement': st.text_area("Announcement"),
            'impact': st.text_area("Business Impact"),
            'future_implications': st.text_area("Future Implications"),
            'stakeholder_benefits': st.text_area("Stakeholder Benefits")
        })
    
    elif post_type == "customer_success":
        form_data.update({
            'customer_name': st.text_input("Customer Name"),
            'industry': st.text_input("Industry"),
            'challenge': st.text_area("Customer Challenge"),
            'solution': st.text_area("Solution Provided"),
            'results': st.text_area("Results Achieved"),
            'testimonial': st.text_area("Customer Testimonial")
        })
    
    elif post_type == "promotional_offer":
        form_data.update({
            'offer_details': st.text_area("Offer Details"),
            'discount_amount': st.text_input("Discount Amount"),
            'validity_period': st.text_input("Validity Period"),
            'terms_conditions': st.text_area("Terms and Conditions"),
            'redemption_process': st.text_area("Redemption Process")
        })
    
    return form_data

def run():
    st.markdown("""
    # Social Media Post Generator
    Generate optimized social media posts using AI
    """)

    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False

    # Model selection
    models = get_ollama_models()
    if not models:
        st.warning("Ollama is not running. Make sure to have the Ollama API accessible.")
        return

    selected_model = st.selectbox("Select AI Model", models, format_func=lambda x: f'🔮 {x}')

    # Post type selection
    post_type = st.selectbox(
        "Select Post Type",
        list(POST_TYPES.keys()),
        format_func=lambda x: POST_TYPES[x]['name']
    )

    # Platform selection
    platforms = st.multiselect(
        "Select Platforms",
        list(PLATFORMS.keys()),
        format_func=lambda x: PLATFORMS[x]['name']
    )

    # Create form
    with st.form("social_media_form"):
        st.subheader(f"Details for {POST_TYPES[post_type]['name']}")
        
        # Render conditional form based on post type
        form_data = render_conditional_form(post_type)
        
        # Tone and style preferences
        st.subheader("Content Preferences")
        form_data['tone'] = st.select_slider(
            "Tone of Voice",
            options=["Professional", "Casual", "Enthusiastic", "Informative", "Humorous"]
        )
        form_data['emoji_usage'] = st.slider(
            "Emoji Usage",
            min_value=0,
            max_value=5,
            value=2,
            help="0 = No emojis, 5 = Maximum emoji usage"
        )
        
        submitted = st.form_submit_button("Generate Posts")
        
        if submitted and platforms:
            st.session_state.form_submitted = True
            st.session_state.form_data = form_data
            st.session_state.post_type = post_type
            st.session_state.platforms = platforms

    # Generate posts if form was submitted
    if st.session_state.form_submitted:
        try:
            llm = Ollama(
                model=selected_model,
                temperature=0.7,  # Higher temperature for more creative content
                base_url=DEFAULT_API_URL,
                auth=(DEFAULT_USERNAME, DEFAULT_PASSWORD),
            )

            st.subheader("Generated Posts")
            
            for platform in st.session_state.platforms:
                with st.expander(f"{PLATFORMS[platform]['name']} Post"):
                    with st.spinner(f"Generating {PLATFORMS[platform]['name']} post..."):
                        prompt = get_post_prompt(
                            st.session_state.post_type,
                            platform,
                            st.session_state.form_data
                        )
                        post_content = llm.predict(prompt)
                        
                        # Display post
                        st.markdown("### Post Preview")
                        st.markdown(post_content)
                        
                        # Add copy button
                        st.button(
                            "Copy to Clipboard",
                            key=f"copy_{platform}",
                            on_click=lambda: st.write("Post copied to clipboard!")
                        )
                        
                        # Display platform-specific metrics
                        st.markdown("### Post Metrics")
                        chars = len(post_content)
                        words = len(post_content.split())
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Characters", chars)
                        with col2:
                            st.metric("Words", words)
                        with col3:
                            st.metric("Chars Remaining", PLATFORMS[platform]['max_length'] - chars)

        except Exception as e:
            st.error(f"Error generating posts: {str(e)}")

        # Add scheduling option
        st.subheader("Schedule Posts")
        col1, col2 = st.columns(2)
        with col1:
            schedule_date = st.date_input("Select Date")
        with col2:
            schedule_time = st.time_input("Select Time")
        
        if st.button("Schedule Posts"):
            st.success(f"Posts scheduled for {schedule_date} at {schedule_time}")

if __name__ == "__main__":
    run()
