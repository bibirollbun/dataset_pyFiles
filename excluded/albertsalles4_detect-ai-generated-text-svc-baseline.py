import pandas as pd
train = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v1.csv")

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train['text'])

from sklearn.svm import SVC
classifier = SVC(probability=True)
classifier.fit(X_train, train['label'])

test   = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
X_test = vectorizer.transform(test['text'])
y_pred = classifier.predict_proba(X_test)[:,1]

sample_submission = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")
sample_submission["id"] = test["id"]
sample_submission["generated"] = y_pred


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, classifier)


print(c.predict_proba(["Hi"]))


from lime.lime_text import LimeTextExplainer
class_names = ["human", "ai"]
explainer = LimeTextExplainer(class_names=class_names)


idx = 15866
text = train["text"].iloc[idx]
label = class_names[train["label"].iloc[idx]]
exp = explainer.explain_instance(text, c.predict_proba, num_features=30)
print(f"Document id: {idx}")
print(f"P(ai) = {c.predict_proba([text])[0,1]}")
print(f"True class: {label}")


text = """
When developing an integration strategy, it's crucial to create a cohesive yet adaptable approach that ensures smooth connectivity between internal tools while remaining compatible with third-party systems and external datasets. The goal is to balance standardization, scalability, and modularity to meet the diverse needs of creating, testing, and validating CCAM systems.
 
Component Integration Approach
 
A well-structured system starts with a modular architecture. You should set clear limits and connections for key components like scenario management systems, marketplaces, and authentication mechanisms. By compartmentalizing responsibilities, each module can operate independently while still complementing the broader system. This not only makes updates easier but also minimizes disruptions when a single module is modified or enhanced.
 
For communication protocols, consider using commonly recognized ones like HTTP/HTTPS for simplicity, or gRPC and MQTT if higher performance or real-time messaging is needed. These standards help ensure smooth data exchange with external systems. Organize data workflows into straightforward pipelines: start with raw data ingestion, process it into intermediate formats, and refine it into validated outputs. Always include metadata to maintain traceability and boost data reliability.
 
Remember, an effective platform should be flexible enough to accommodate dynamic workflows. Users should be able to design and execute customized processes using modular tools and datasets. Offering both on-premise and cloud-based options provides operational flexibility and efficiency across various user scenarios.
 
Interoperability Guidelines
 
Ensuring interoperability is essential for creating a cohesive system that works internally and with external ecosystems. Use established standards and shared frameworks for interpreting and exchanging data to reduce friction and promote inclusivity among stakeholders.
 
If your system needs to access distributed datasets, implementing a federated framework can be beneficial. A centralized access point simplifies data retrieval while maintaining efficiency and reliability. Avoid relying on numerous individual API connections that can complicate things and cause performance issues.
 
Pay attention to how data is structured. Curate ontologies and metadata carefully to ensure consistency across tools and datasets. This practice reduces errors and makes your system more robust for concurrent operations. Additionally, build a versioning mechanism to manage updates without disrupting workflows. Users should trust that improvements won’t cause unnecessary complications.
 
Scalability and Performance
 
Designing for scalability is non-negotiable when handling CCAM-related tasks. Plan for high data volumes and demanding processing requirements by adopting distributed storage and efficient data pipelines. These techniques help reduce latency and ensure smooth data flows.
 
Optimize performance by tailoring the execution environment to user needs. Whether processes run locally or in the cloud, flexibility ensures your system can adapt to diverse operational contexts. Balance resources wisely to achieve the best performance in every situation.
 
Testing and Validation Guidelines
 
No integration strategy is complete without thorough testing and validation procedures. Start by testing each component individually to ensure it meets its intended purpose. Only after individual modules are validated should you move on to system-wide testing to confirm seamless interaction between components.
 
To future-proof your platform, conduct regular cross-compatibility tests with external tools and datasets. This step ensures ongoing interoperability across the ecosystem. Don’t overlook continuous monitoring; your system should include mechanisms to detect and adapt to potential issues as they arise. A proactive approach not only ensures stability over time but also builds user trust in your platform’s reliability.
"""


exp = explainer.explain_instance(text, c.predict_proba, num_features=200)
exp.as_list()


%matplotlib inline
fig = exp.as_pyplot_figure()


exp.show_in_notebook(text=True)







