# Finance-RAG
# GTE Finance Model Deployment on Azure Machine Learning Workspace

This guide walks you through deploying the GTE Finance embedding model as a managed endpoint on Azure Machine Learning. The model is based on sentence transformers and is optimized for financial text embeddings.

## Prerequisites

- Azure subscription with sufficient credits
- Azure CLI installed and configured
- Git installed
- Appropriate permissions to create Azure resources

## Architecture Overview

The deployment creates:
- Azure Resource Group
- Azure Machine Learning Workspace
- Compute Instance for development
- Managed Online Endpoint for model serving
- Custom environment with sentence transformers

## Step 1: Initial Setup in Azure Cloud Shell

### 1.1 Open Azure Cloud Shell
1. Navigate to [Azure Portal](https://portal.azure.com)
2. Click on the Cloud Shell icon (terminal icon) in the top navigation bar
3. Select **Bash** when prompted

### 1.2 Clone the Repository
```bash
# Clone the repository and create/switch to the data-ingestion-pipeline branch
git clone https://github.com/Chandi713/Finance-RAG.git checkout -b "data-ingestion-pipeline"
cd checkout
```

### 1.3 Run Setup Script
```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script to create Azure resources
./setup.sh
```

**What the setup script does:**
- Generates a unique suffix for resource naming
- Registers required Azure resource providers
- Creates a resource group
- Creates an Azure Machine Learning workspace
- Creates a compute instance
- Sets default configurations for Azure CLI

⏱️ **Wait Time:** Allow 10-15 minutes for all resources to be provisioned.

## Step 2: Access Azure Machine Learning Workspace

### 2.1 Navigate to ML Workspace
1. In the Azure Portal, search for "Machine Learning"
2. Click on your newly created workspace (format: `financerag-mlw{suffix}`)
3. Click "Launch studio" to open Azure ML Studio

### 2.2 Access Compute Instance
1. In Azure ML Studio, go to **Compute** in the left navigation
2. Click on **Compute instances** tab
3. Find your compute instance (format: `ci{suffix}`)
4. Wait for the status to show **Running** (this may take a few minutes)
5. Click on **Terminal** to open the Jupyter terminal

## Step 3: Model Deployment Setup

### 3.1 Clone Repository in Compute Instance
In the compute instance terminal, run:

```bash
# Clone the repository again in the compute instance
git clone https://github.com/Chandi713/Finance-RAG.git checkout -b "data-ingestion-pipeline"

```

### 3.2 Verify Model Files
The model files are already included in the repository. Verify the model directory structure:
```
model/
├── gte-finance-model/
│   ├── 1_Pooling/
│   ├── README.md
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── config_sentence_transformers.json
│   ├── modules.json
│   ├── sentence_bert_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json

```

✅ **Note**: All required model files are pre-included in the repository, so no additional model preparation is needed.

## Step 4: Run the Deployment Notebook

### 4.1 Open the Notebook
1. In the compute instance, navigate to the **Notebooks** section
2. Find and open `ManagedEndpoint.ipynb`
3. Ensure the kernel is set to **Python 3.10 - AzureML**

### 4.2 Execute the Notebook
Run each cell in sequence. The notebook will:

1. **Install Dependencies**: Install `azure-ai-ml` package
2. **Authentication**: Set up Azure credentials
3. **Initialize ML Client**: Connect to your workspace
4. **Create Endpoint**: Set up the managed online endpoint
5. **Register Model**: Upload the pre-included GTE Finance model from the repository
6. **Create Environment**: Set up the conda environment with required packages
7. **Create Deployment**: Deploy the model with specified compute resources
8. **Allocate Traffic**: Route 100% traffic to the new deployment
9. **Test Deployment**: Verify the endpoint works correctly

✅ **Important**: The model is already included in the repository's `model/` directory, so the registration step will upload the existing model files.

### 4.3 Key Configuration Parameters

The deployment uses these settings:
- **Instance Type**: `Standard_E4s_v3` (4 cores, 32GB RAM)
- **Instance Count**: 2 (for high availability)
- **Request Timeout**: 180 seconds
- **Max Sequence Length**: 2000 tokens
- **Batch Size**: 32 (configurable in score.py)

## Step 5: Model Testing

### 5.1 Test the Endpoint
The notebook includes a test section that sends sample text to the endpoint. You can modify the test data:

```python
data = {
    "texts": [
        "Your financial text here",
        "Another document to embed"
    ]
}
```

### 5.2 Expected Response Format
```json
{
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "dimensions": 768,
    "count": 2
}
```

## Step 6: Production Usage

### 6.1 Get Endpoint Details
After successful deployment, note:
- **Endpoint URL**: Found in the Azure ML Studio under Endpoints
- **Authentication Key**: Available in the endpoint's authentication section

### 6.2 Making API Calls
```python
import requests
import json

url = "https://your-endpoint-url/score"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-api-key"
}

data = {
    "texts": ["Financial text to embed"]
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
```

## Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**
   - Verify model files are present in the correct directory
   - Check that the Hugging Face model ID is correct: `Yaksh170802/gte-finance-model`

2. **Environment Setup Issues**
   - Ensure all dependencies in `environment.yml` are compatible
   - Check that the base image supports the required packages

3. **Deployment Timeout**
   - Increase the request timeout in deployment settings
   - Monitor the deployment logs for specific error messages

4. **Authentication Errors**
   - Verify your Azure credentials are correctly configured
   - Ensure you have sufficient permissions on the subscription

### Monitoring and Logs

1. **View Deployment Logs**:
   - Go to Endpoints in Azure ML Studio
   - Select your endpoint and deployment
   - Check the "Logs" tab for detailed information

2. **Monitor Performance**:
   - Use the "Metrics" tab to monitor request latency and success rates
   - Set up alerts for failures or high latency

## Cleanup

To avoid ongoing charges, you can delete resources:

```bash
# Delete the entire resource group (this removes all resources)
az group delete --name financerag-rg{your-suffix} --yes --no-wait
```

## Configuration Details

### Model Specifications
- **Base Model**: GTE (General Text Embeddings) fine-tuned for finance
- **Embedding Dimensions**: 768
- **Max Sequence Length**: 2000 tokens
- **Normalization**: Enabled by default

### Compute Resources
- **Development**: Standard_DS11_V2 compute instance
- **Production**: Standard_E4s_v3 (2 instances for HA)

### Security
- **Authentication**: Key-based authentication
- **Network**: Public endpoint (can be configured for private)

## Support

For issues related to:
- **Azure ML**: Check Azure ML documentation or support
- **Model**: Refer to the sentence-transformers documentation
- **Deployment**: Review the logs and error messages in Azure ML Studio


