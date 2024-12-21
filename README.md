# Customer-Service-Assistant-App---Google-Gemini

## Problem Statement : Build a Web application using streamlit framework for Customer Service Assistant . The search prompts are text only

![image](https://github.com/user-attachments/assets/20bdf07e-38e0-43f8-ad3e-b1929284388f)

## Application code explanation : app.py
   -  Import necessary dependencies
   -  Define the utility function for Data chunking
   -  Define the utility function to process the PDF
   -  Define the utility function to generate text embeddings(leveraged google's 'embedding-001' model )
   -  Define the text retrieval utility function using RAG
   -  Set the application title and headers
   -  Set the environmental variables & validate if loaded properly
   -  Upload a PDF file and read its content using PyPDF2.
   -  Peform data chunking and embedding using the utility functions
   -  Create a chromadb client and store the document and embeddings
   -  Get user question from application interface
   -  Retrive relevant documents and show in the application interface
   -  Design the sidebar and footer

## Application Demo 

### 1. how the interface looks like 

![image](https://github.com/user-attachments/assets/4c9449d4-ff20-499c-b1dd-f7ef24434245)

### 2. Query / Output combination 

a. What are the details of the Yeezy Boost 350 V2

![image](https://github.com/user-attachments/assets/c732b87e-6dc7-4e27-95d3-1fd38054e0cd)

b.  What are the details of the Puma Suede Classic XXI

![image](https://github.com/user-attachments/assets/416af10c-7198-4a54-bbee-b9a392fbf382)

c. What are the details of the PumaCali Dream Women

![image](https://github.com/user-attachments/assets/08d2e8d0-9235-436b-819d-5b9c9f8aa213)

d. What are the details of the Nike Air Force 1 Low By You Women

![image](https://github.com/user-attachments/assets/024416f1-7bdc-48ac-803e-7709fe13623c)

e. Search by different text : 

![image](https://github.com/user-attachments/assets/af6443b5-17ea-4be3-a8a2-d5d890f1c694)

![image](https://github.com/user-attachments/assets/9ece5278-7608-430e-814a-90d482d75d57)

![image](https://github.com/user-attachments/assets/e576c792-e8f5-4a04-b185-60f7ee9a6877)

![image](https://github.com/user-attachments/assets/d4140651-d4c4-4901-981a-ead1064d38e7)

![image](https://github.com/user-attachments/assets/19805fc4-730f-430e-8484-02c9e4028588)

![image](https://github.com/user-attachments/assets/23022f51-75e6-4334-a156-c2fbdcfc3d86)

![image](https://github.com/user-attachments/assets/cf18450e-929e-41b6-94de-4dbccecadba8)

![image](https://github.com/user-attachments/assets/7691ef03-df8f-483a-bd1e-d9c0c175db6b)

## Steps of implementation

   1. Step1 : Setup the folder structure
   2. Step2 : Add the Google API Key in the .env file
   3. Step3 : Install necessary dependencies from the terminal ( command : pip install -r requirements.txt )
   4. Step4 : Validate the app.py file from the terminal ( command : python app.py )
   5. Step5 : Visualize the web application ( command streamlit run app.py )


Happy Coding :)














