import os
import streamlit as st
from haystack import Document
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore # Internal Vector Storage like ChromaDB in LangChain
from haystack.nodes.retriever.multimodal import MultiModalRetriever

class MultimodalSearch:
    def __init__(
                self,
                doc_dir = "fashon-db"
                ):
        self.document_store = InMemoryDocumentStore(embedding_dim=512) # store 512 dim image embeddings

        images = [
                Document(content=f"./{doc_dir}/{filename}", content_type="image")
                for filename in os.listdir(f"./{doc_dir}")
                ]

        self.document_store.write_documents(images)

        self.retriever_text_to_image = MultiModalRetriever(
                                                            document_store=self.document_store,
                                                            query_embedding_model="sentence-transformers/clip-ViT-B-32",
                                                            query_type="text",
                                                            document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
                                                            )

        # Turn images into embeddings and store them in the DocumentStore
        self.document_store.update_embeddings(retriever=self.retriever_text_to_image)

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=self.retriever_text_to_image, name="retriever_text_to_image", inputs=["Query"])

    def search(self, query, top_k=3):
        results = self.pipeline.run(query=query, params={"retriever_text_to_image": {"top_k": top_k}})
        return sorted(results["documents"], key=lambda d: d.score, reverse=True)
    
st.set_page_config(
    layout="wide"
)

def main():
    st.markdown("<h1 style='text-align: center; color: green;'>Fashion Search Store</h1>", unsafe_allow_html=True)

    multimodal_search = MultimodalSearch()

    query = st.text_input("Enter your cloth requirements:")
    if st.button("Search"):
        if len(query) > 0:
            results = multimodal_search.search(query)
            st.warning("Your query was "+query)
            st.subheader("Search Results:")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.write(f"Score: {round(results[0].score*100, 2)}%")
                st.image(results[0].content, use_column_width=True)
            with col2:
                st.write(f"Score: {round(results[1].score*100, 2)}%")
                st.image(results[1].content, use_column_width=True)
            with col3:
                st.write(f"Score: {round(results[2].score*100, 2)}%")
                st.image(results[2].content, use_column_width=True)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()