"""
Integration script to add ML capabilities to the SCEF web interface.
Run this script to modify app.py to include ML components.
"""

import os
import sys
import re


def modify_app_py():
    """
    Modify app.py to include ML components.
    """
    app_py_path = "web/app.py"
    
    if not os.path.exists(app_py_path):
        print(f"Error: {app_py_path} not found. Make sure you're running from the SCEF root directory.")
        return False
    
    with open(app_py_path, 'r') as f:
        content = f.read()
    
    # Check if ML components are already included
    if "from web.ml_api import include_ml_router" in content:
        print("ML components are already included in app.py")
        return True
    
    # Add imports
    import_pattern = "# Import SCEF components"
    if import_pattern in content:
        ml_imports = """# Import SCEF components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.ml_api import include_ml_router
"""
        content = content.replace(import_pattern, ml_imports)
    else:
        print("Warning: Could not find import section in app.py")
        
        # Add imports at the top of the file
        ml_imports = """import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.ml_api import include_ml_router

"""
        content = ml_imports + content
    
    # Add ML router initialization
    app_creation_pattern = "app = FastAPI("
    if app_creation_pattern in content:
        # Find the end of the app creation section
        lines = content.split('\n')
        app_line_index = -1
        
        for i, line in enumerate(lines):
            if app_creation_pattern in line:
                app_line_index = i
                break
        
        if app_line_index >= 0:
            # Find a good place to insert the ML router inclusion
            # Look for the first line after app creation that's not indented
            insert_index = app_line_index + 1
            
            while insert_index < len(lines) and (
                lines[insert_index].strip() == '' or 
                lines[insert_index].startswith(' ') or 
                lines[insert_index].startswith('\t')
            ):
                insert_index += 1
            
            # Insert the ML router inclusion
            lines.insert(insert_index, "# Include ML router")
            lines.insert(insert_index + 1, "include_ml_router(app)")
            lines.insert(insert_index + 2, "")
            
            content = '\n'.join(lines)
        else:
            print("Warning: Could not find app creation in app.py")
    else:
        print("Warning: Could not find app creation in app.py")
    
    # Write modified content
    with open(app_py_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully modified {app_py_path} to include ML components")
    return True


def create_web_ml_templates():
    """
    Create ML-related templates for the web interface.
    """
    templates_dir = "web/templates/ml"
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create ML templates
    ml_templates = {
        "ml_index.html": """{% extends "base.html" %}

{% block title %}SCEF - Machine Learning{% endblock %}

{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
    <h1>Machine Learning</h1>
    <div>
        <a href="/ml/models" class="btn btn-primary">
            <i class="bi bi-diagram-3"></i> ML Models
        </a>
        <a href="/ml/strategies" class="btn btn-success">
            <i class="bi bi-robot"></i> ML Strategies
        </a>
    </div>
</div>

<div class="row">
    <div class="col-md-12">
        <div class="card mb-4">
            <div class="card-header">
                <h5>Machine Learning Components</h5>
            </div>
            <div class="card-body">
                <p>The SCEF Machine Learning module provides tools for creating and testing ML-based trading strategies.</p>
                
                <div class="row mt-4">
                    <div class="col-md-4 mb-3">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title"><i class="bi bi-grid-3x3-gap"></i> Feature Engineering</h5>
                                <p class="card-text">Extract meaningful features from market data for ML models.</p>
                                <a href="/ml/features" class="btn btn-outline-primary">Explore Features</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4 mb-3">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title"><i class="bi bi-cpu"></i> ML Models</h5>
                                <p class="card-text">Create, train, and test machine learning models for prediction.</p>
                                <a href="/ml/models" class="btn btn-outline-primary">Manage Models</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4 mb-3">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title"><i class="bi bi-robot"></i> ML Strategies</h5>
                                <p class="card-text">Build trading strategies powered by ML models and RL agents.</p>
                                <a href="/ml/strategies" class="btn btn-outline-primary">Create Strategies</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-6">
        <div class="card mb-4">
            <div class="card-header">
                <h5>Getting Started</h5>
            </div>
            <div class="card-body">
                <ol>
                    <li>Upload market data from the <a href="/data">Market Data</a> page</li>
                    <li>Create and train ML models on the <a href="/ml/models">ML Models</a> page</li>
                    <li>Build ML-based strategies on the <a href="/ml/strategies">ML Strategies</a> page</li>
                    <li>Backtest your strategies on the <a href="/backtest">Backtest</a> page</li>
                    <li>Deploy to production from the <a href="/deploy">Deployment</a> page</li>
                </ol>
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        <div class="card mb-4">
            <div class="card-header">
                <h5>Available Techniques</h5>
            </div>
            <div class="card-body">
                <div class="list-group">
                    <a href="/ml/models?type=classifier" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                        Classification Models
                        <span class="badge bg-primary rounded-pill">Random Forest, SVM, Logistic Regression</span>
                    </a>
                    <a href="/ml/models?type=regressor" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                        Regression Models
                        <span class="badge bg-primary rounded-pill">Linear, Random Forest, SVR</span>
                    </a>
                    <a href="/ml/strategies?type=rl" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                        Reinforcement Learning
                        <span class="badge bg-primary rounded-pill">DQN, Policy Gradients</span>
                    </a>
                    <a href="/ml/strategies?type=ensemble" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                        Ensemble Methods
                        <span class="badge bg-primary rounded-pill">Weighted Avg, Voting, Stacking</span>
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}""",

        "ml_models.html": """{% extends "base.html" %}

{% block title %}SCEF - ML Models{% endblock %}

{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
    <h1>ML Models</h1>
    <div>
        <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#createModelModal">
            <i class="bi bi-plus-circle"></i> Create Model
        </button>
    </div>
</div>

<div class="row">
    <div class="col-md-12">
        <div class="card mb-4">
            <div class="card-header">
                <h5>Available Models</h5>
            </div>
            <div class="card-body">
                <div id="models-list" class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Type</th>
                                <th>Status</th>
                                <th>Created</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="models-table-body">
                            <!-- Models will be loaded here -->
                            <tr>
                                <td colspan="5" class="text-center">Loading models...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Create Model Modal -->
<div class="modal fade" id="createModelModal" tabindex="-1" aria-labelledby="createModelModalLabel" aria-hidden="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="createModelModalLabel">Create ML Model</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <form id="createModelForm">
                    <div class="mb-3">
                        <label for="model-type" class="form-label">Model Type</label>
                        <select class="form-control" id="model-type" name="model_type" required>
                            <option value="">Select model type...</option>
                            <option value="classifier">Classifier</option>
                            <option value="regressor">Regressor</option>
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="model-name" class="form-label">Model Algorithm</label>
                        <select class="form-control" id="model-name" name="model_name" required>
                            <option value="">Select algorithm...</option>
                            <optgroup label="Classifiers" id="classifier-options">
                                <option value="random_forest">Random Forest</option>
                                <option value="logistic_regression">Logistic Regression</option>
                                <option value="svm">Support Vector Machine</option>
                            </optgroup>
                            <optgroup label="Regressors" id="regressor-options">
                                <option value="random_forest">Random Forest</option>
                                <option value="linear_regression">Linear Regression</option>
                                <option value="svm">Support Vector Regression</option>
                            </optgroup>
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="prediction-horizon" class="form-label">Prediction Horizon</label>
                        <input type="number" class="form-control" id="prediction-horizon" name="prediction_horizon" value="1" min="1" max="20" required>
                        <div class="form-text">Number of periods ahead to predict.</div>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="button" class="btn btn-primary" id="create-model-btn">Create</button>
            </div>
        </div>
    </div>
</div>

<!-- Train Model Modal -->
<div class="modal fade" id="trainModelModal" tabindex="-1" aria-labelledby="trainModelModalLabel" aria-hidden="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="trainModelModalLabel">Train Model</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <form id="trainModelForm">
                    <input type="hidden" id="train-model-id" name="model_id">
                    
                    <div class="mb-3">
                        <label for="data-select" class="form-label">Market Data</label>
                        <select class="form-control" id="data-select" name="data_id" required>
                            <option value="">Select market data...</option>
                            <!-- Market data will be loaded here -->
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="train-test-split" class="form-label">Train/Test Split</label>
                        <input type="range" class="form-range" id="train-test-split" name="train_test_split" min="0.5" max="0.9" step="0.05" value="0.7">
                        <div class="d-flex justify-content-between">
                            <span>50% Train</span>
                            <span id="split-value">70% Train</span>
                            <span>90% Train</span>
                        </div>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="button" class="btn btn-primary" id="train-model-btn">Train</button>
            </div>
        </div>
    </div>
</div>

<!-- Model Details Modal -->
<div class="modal fade" id="modelDetailsModal" tabindex="-1" aria-labelledby="modelDetailsModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="modelDetailsModalLabel">Model Details</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <div id="model-details-content">
                    <!-- Model details will be loaded here -->
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                <button type="button" class="btn btn-success" id="create-strategy-btn">Create Strategy</button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Load models
        loadModels();
        
        // Load market data for training
        loadMarketData();
        
        // Initialize modals
        const createModelModal = new bootstrap.Modal(document.getElementById('createModelModal'));
        const trainModelModal = new bootstrap.Modal(document.getElementById('trainModelModal'));
        const modelDetailsModal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
        
        // Handle model type selection
        document.getElementById('model-type').addEventListener('change', function() {
            const modelType = this.value;
            
            // Show/hide algorithm options based on model type
            document.querySelectorAll('#model-name optgroup').forEach(function(optgroup) {
                optgroup.style.display = 'none';
            });
            
            if (modelType === 'classifier') {
                document.getElementById('classifier-options').style.display = 'block';
            } else if (modelType === 'regressor') {
                document.getElementById('regressor-options').style.display = 'block';
            }
            
            // Reset model name selection
            document.getElementById('model-name').value = '';
        });
        
        // Handle train/test split slider
        document.getElementById('train-test-split').addEventListener('input', function() {
            const value = this.value;
            document.getElementById('split-value').textContent = `${Math.round(value * 100)}% Train`;
        });
        
        // Create model button
        document.getElementById('create-model-btn').addEventListener('click', createModel);
        
        // Train model button
        document.getElementById('train-model-btn').addEventListener('click', trainModel);
    });
    
    function loadModels() {
        fetch('/api/ml/models')
            .then(response => response.json())
            .then(models => {
                const tableBody = document.getElementById('models-table-body');
                
                if (models.length === 0) {
                    tableBody.innerHTML = '<tr><td colspan="5" class="text-center">No models found. Create your first model!</td></tr>';
                    return;
                }
                
                tableBody.innerHTML = '';
                
                models.forEach(model => {
                    const row = document.createElement('tr');
                    
                    row.innerHTML = `
                        <td>${model.model_name || model.id.substring(0, 8)}</td>
                        <td>${model.model_type}</td>
                        <td>${model.is_trained ? '<span class="badge bg-success">Trained</span>' : '<span class="badge bg-warning">Not Trained</span>'}</td>
                        <td>${new Date(model.created_at).toLocaleString()}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary view-model-btn" data-model-id="${model.id}">View</button>
                            <button class="btn btn-sm btn-outline-success train-model-btn" data-model-id="${model.id}" ${model.is_trained ? 'disabled' : ''}>Train</button>
                            <button class="btn btn-sm btn-outline-danger create-strategy-from-model-btn" data-model-id="${model.id}" ${!model.is_trained ? 'disabled' : ''}>Create Strategy</button>
                        </td>
                    `;
                    
                    tableBody.appendChild(row);
                });
                
                // Add event listeners to buttons
                document.querySelectorAll('.view-model-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const modelId = this.getAttribute('data-model-id');
                        viewModel(modelId);
                    });
                });
                
                document.querySelectorAll('.train-model-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const modelId = this.getAttribute('data-model-id');
                        openTrainModelModal(modelId);
                    });
                });
                
                document.querySelectorAll('.create-strategy-from-model-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const modelId = this.getAttribute('data-model-id');
                        createStrategyFromModel(modelId);
                    });
                });
            })
            .catch(error => {
                console.error('Error loading models:', error);
                document.getElementById('models-table-body').innerHTML = 
                    `<tr><td colspan="5" class="text-center text-danger">Error loading models: ${error}</td></tr>`;
            });
    }
    
    function loadMarketData() {
        fetch('/api/data')
            .then(response => response.json())
            .then(data => {
                const selectElement = document.getElementById('data-select');
                
                // Keep the first option (placeholder)
                const placeholder = selectElement.options[0];
                selectElement.innerHTML = '';
                selectElement.appendChild(placeholder);
                
                data.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.id;
                    option.textContent = item.name;
                    selectElement.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error loading market data:', error);
                showToast('Error loading market data: ' + error, 'danger');
            });
    }
    
    function createModel() {
        const form = document.getElementById('createModelForm');
        
        // Validate form
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        
        // Get form values
        const modelType = document.getElementById('model-type').value;
        const modelName = document.getElementById('model-name').value;
        const predictionHorizon = parseInt(document.getElementById('prediction-horizon').value);
        
        // Create model config
        const modelConfig = {
            model_type: modelType,
            model_name: modelName,
            prediction_horizon: predictionHorizon
        };
        
        // Send request to create model
        fetch('/api/ml/models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(modelConfig)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Show success message
                showToast('Model created successfully!', 'success');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('createModelModal'));
                modal.hide();
                
                // Reset form
                form.reset();
                
                // Reload models
                loadModels();
                
                // Open train modal for the new model
                openTrainModelModal(data.model_id);
            } else {
                showToast('Error creating model: ' + (data.message || 'Unknown error'), 'danger');
            }
        })
        .catch(error => {
            console.error('Error creating model:', error);
            showToast('Error creating model: ' + error, 'danger');
        });
    }
    
    function openTrainModelModal(modelId) {
        // Set model ID in form
        document.getElementById('train-model-id').value = modelId;
        
        // Open modal
        const modal = new bootstrap.Modal(document.getElementById('trainModelModal'));
        modal.show();
    }
    
    function trainModel() {
        const form = document.getElementById('trainModelForm');
        
        // Validate form
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        
        // Get form values
        const modelId = document.getElementById('train-model-id').value;
        const dataId = document.getElementById('data-select').value;
        const trainTestSplit = parseFloat(document.getElementById('train-test-split').value);
        
        // Create request data
        const requestData = {
            model_id: modelId,
            data_id: dataId,
            train_test_split: trainTestSplit,
            model_config: null, // Not needed since we're using an existing model
            feature_params: {
                windows: [5, 10, 20, 50],
                include_time_features: true,
                lookback: 1,
                normalization_method: "z_score"
            }
        };
        
        // Disable button and show loading state
        const button = document.getElementById('train-model-btn');
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...';
        
        // Send request to train model
        fetch('/api/ml/models/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Show success message
                showToast('Model trained successfully!', 'success');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('trainModelModal'));
                modal.hide();
                
                // Reset form
                form.reset();
                
                // Reload models
                loadModels();
                
                // Show model details
                viewModel(modelId);
            } else {
                showToast('Error training model: ' + (data.message || 'Unknown error'), 'danger');
            }
        })
        .catch(error => {
            console.error('Error training model:', error);
            showToast('Error training model: ' + error, 'danger');
        })
        .finally(() => {
            // Reset button
            button.disabled = false;
            button.innerHTML = 'Train';
        });
    }
    
    function viewModel(modelId) {
        fetch(`/api/ml/models/${modelId}`)
            .then(response => response.json())
            .then(model => {
                const detailsContainer = document.getElementById('model-details-content');
                
                detailsContainer.innerHTML = `
                    <div class="mb-3">
                        <h4>${model.config.model_name} ${model.config.model_type}</h4>
                        <p><strong>Status:</strong> ${model.is_trained ? '<span class="badge bg-success">Trained</span>' : '<span class="badge bg-warning">Not Trained</span>'}</p>
                        <p><strong>Created:</strong> ${new Date(model.created_at).toLocaleString()}</p>
                        <p><strong>Prediction Horizon:</strong> ${model.config.prediction_horizon} periods</p>
                    </div>
                    
                    <div class="mb-3">
                        <h5>Model Configuration</h5>
                        <pre class="bg-light p-3 rounded">${JSON.stringify(model.config, null, 2)}</pre>
                    </div>
                `;
                
                // Show modal
                const modal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
                modal.show();
                
                // Update create strategy button
                const createStrategyBtn = document.getElementById('create-strategy-btn');
                createStrategyBtn.disabled = !model.is_trained;
                createStrategyBtn.onclick = function() {
                    createStrategyFromModel(modelId);
                    modal.hide();
                };
            })
            .catch(error => {
                console.error('Error loading model details:', error);
                showToast('Error loading model details: ' + error, 'danger');
            });
    }
    
    function createStrategyFromModel(modelId) {
        // Redirect to ML strategies page with model ID
        window.location.href = `/ml/strategies?model=${modelId}`;
    }
</script>
{% endblock %}""",

        "ml_strategies.html": """{% extends "base.html" %}

{% block title %}SCEF - ML Strategies{% endblock %}

{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
    <h1>ML Strategies</h1>
    <div>
        <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#createMLStrategyModal">
            <i class="bi bi-plus-circle"></i> Create ML Strategy
        </button>
        <button class="btn btn-success" data-bs-toggle="modal" data-bs-target="#createRLStrategyModal">
            <i class="bi bi-plus-circle"></i> Create RL Strategy
        </button>
        <button class="btn btn-info" data-bs-toggle="modal" data-bs-target="#createEnsembleStrategyModal">
            <i class="bi bi-plus-circle"></i> Create Ensemble
        </button>
    </div>
</div>

<div class="row">
    <div class="col-md-12">
        <div class="card mb-4">
            <div class="card-header">
                <h5>Available Strategies</h5>
            </div>
            <div class="card-body">
                <div id="strategies-list" class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Type</th>
                                <th>Created</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="strategies-table-body">
                            <!-- Strategies will be loaded here -->
                            <tr>
                                <td colspan="4" class="text-center">Loading strategies...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Create ML Strategy Modal -->
<div class="modal fade" id="createMLStrategyModal" tabindex="-1" aria-labelledby="createMLStrategyModalLabel" aria-hidden="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="createMLStrategyModalLabel">Create ML Strategy</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <form id="createMLStrategyForm">
                    <div class="mb-3">
                        <label for="ml-strategy-name" class="form-label">Strategy Name</label>
                        <input type="text" class="form-control" id="ml-strategy-name" name="name" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="ml-model-select" class="form-label">ML Model</label>
                        <select class="form-control" id="ml-model-select" name="model_id" required>
                            <option value="">Select ML model...</option>
                            <!-- ML models will be loaded here -->
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="ml-position-sizing" class="form-label">Position Sizing</label>
                        <div class="input-group">
                            <input type="number" class="form-control" id="ml-position-sizing" name="position_sizing" value="1.0" min="0.1" max="10" step="0.1" required>
                            <span class="input-group-text">x</span>
                        </div>
                        <div class="form-text">Maximum position size as a multiplier of capital.</div>
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label">Risk Control</label>
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="use-volatility-scaling" name="use_volatility_scaling" value="true">
                            <label class="form-check-label" for="use-volatility-scaling">
                                Use Volatility Scaling
                            </label>
                        </div>
                    </div>
                    
                    <div id="volatility-scaling-options" style="display: none;">
                        <div class="mb-3">
                            <label for="volatility-lookback" class="form-label">Volatility Lookback</label>
                            <input type="number" class="form-control" id="volatility-lookback" name="volatility_lookback" value="20" min="5" max="100" step="1">
                            <div class="form-text">Number of periods to use for volatility calculation.</div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="target-volatility" class="form-label">Target Volatility</label>
                            <div class="input-group">
                                <input type="number" class="form-control" id="target-volatility" name="target_volatility" value="0.01" min="0.001" max="0.1" step="0.001">
                                <span class="input-group-text">%</span>
                            </div>
                            <div class="form-text">Target portfolio volatility.</div>
                        </div>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="button" class="btn btn-primary" id="create-ml-strategy-btn">Create</button>
            </div>
        </div>
    </div>
</div>

<!-- Create RL Strategy Modal -->
<div class="modal fade" id="createRLStrategyModal" tabindex="-1" aria-labelledby="createRLStrategyModalLabel" aria-hidden="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="createRLStrategyModalLabel">Create RL Strategy</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <form id="createRLStrategyForm">
                    <div class="mb-3">
                        <label for="rl-strategy-name" class="form-label">Strategy Name</label>
                        <input type="text" class="form-control" id="rl-strategy-name" name="name" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="window-size" class="form-label">Window Size</label>
                        <input type="number" class="form-control" id="window-size" name="window_size" value="20" min="5" max="100" step="1" required>
                        <div class="form-text">Number of periods to include in the state observation.</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="action-space" class="form-label">Action Space</label>
                        <select class="form-control" id="action-space" name="action_space" required>
                            <option value="3">3 (Short/Neutral/Long)</option>
                            <option value="5">5 (Finer Granularity)</option>
                            <option value="7">7 (High Granularity)</option>
                        </select>
                        <div class="form-text">Number of discrete actions for the RL agent.</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="rl-position-sizing" class="form-label">Position Sizing</label>
                        <div class="input-group">
                            <input type="number" class="form-control" id="rl-position-sizing" name="position_sizing" value="1.0" min="0.1" max="10" step="0.1" required>
                            <span class="input-group-text">x</span>
                        </div>
                        <div class="form-text">Maximum position size as a multiplier of capital.</div>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="button" class="btn btn-primary" id="create-rl-strategy-btn">Create</button>
            </div>
        </div>
    </div>
</div>

<!-- Create Ensemble Strategy Modal -->
<div class="modal fade" id="createEnsembleStrategyModal" tabindex="-1" aria-labelledby="createEnsembleStrategyModalLabel" aria-hidden="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="createEnsembleStrategyModalLabel">Create Ensemble Strategy</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <form id="createEnsembleStrategyForm">
                    <div class="mb-3">
                        <label for="ensemble-strategy-name" class="form-label">Strategy Name</label>
                        <input type="text" class="form-control" id="ensemble-strategy-name" name="name" required>
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label">Component Strategies</label>
                        <div id="strategy-checkboxes" class="mb-3">
                            <!-- Strategies will be loaded here -->
                            <div class="text-center">
                                <div class="spinner-border spinner-border-sm" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <span class="ms-2">Loading strategies...</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="ensemble-method" class="form-label">Ensemble Method</label>
                        <select class="form-control" id="ensemble-method" name="ensemble_method" required>
                            <option value="weighted_average">Weighted Average</option>
                            <option value="voting">Voting</option>
                            <option value="dynamic">Dynamic Weighting</option>
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="adaptation-method" class="form-label">Adaptation Method</label>
                        <select class="form-control" id="adaptation-method" name="adaptation_method">
                            <option value="">None</option>
                            <option value="performance">Performance-based</option>
                            <option value="momentum">Momentum-based</option>
                        </select>
                        <div class="form-text">Method for adapting weights over time.</div>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="button" class="btn btn-primary" id="create-ensemble-strategy-btn">Create</button>
            </div>
        </div>
    </div>
</div>

<!-- Strategy Details Modal -->
<div class="modal fade" id="strategyDetailsModal" tabindex="-1" aria-labelledby="strategyDetailsModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="strategyDetailsModalLabel">Strategy Details</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <div id="strategy-details-content">
                    <!-- Strategy details will be loaded here -->
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                <button type="button" class="btn btn-success" id="backtest-strategy-btn">Run Backtest</button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Load strategies
        loadStrategies();
        
        // Load ML models for the ML strategy form
        loadMLModels();
        
        // Initialize modals
        const createMLStrategyModal = new bootstrap.Modal(document.getElementById('createMLStrategyModal'));
        const createRLStrategyModal = new bootstrap.Modal(document.getElementById('createRLStrategyModal'));
        const createEnsembleStrategyModal = new bootstrap.Modal(document.getElementById('createEnsembleStrategyModal'));
        
        // Toggle volatility scaling options
        document.getElementById('use-volatility-scaling').addEventListener('change', function() {
            document.getElementById('volatility-scaling-options').style.display = this.checked ? 'block' : 'none';
        });
        
        // Create ML strategy button
        document.getElementById('create-ml-strategy-btn').addEventListener('click', createMLStrategy);
        
        // Create RL strategy button
        document.getElementById('create-rl-strategy-btn').addEventListener('click', createRLStrategy);
        
        // Create ensemble strategy button
        document.getElementById('create-ensemble-strategy-btn').addEventListener('click', createEnsembleStrategy);
        
        // Check URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        const modelId = urlParams.get('model');
        
        if (modelId) {
            // Pre-select model in the form
            document.getElementById('ml-model-select').value = modelId;
            
            // Open ML strategy modal
            createMLStrategyModal.show();
        }
    });
    
    function loadStrategies() {
        fetch('/api/strategies')
            .then(response => response.json())
            .then(strategies => {
                const tableBody = document.getElementById('strategies-table-body');
                
                if (strategies.length === 0) {
                    tableBody.innerHTML = '<tr><td colspan="4" class="text-center">No strategies found. Create your first strategy!</td></tr>';
                    loadCheckboxes([]);
                    return;
                }
                
                tableBody.innerHTML = '';
                
                const mlStrategies = strategies.filter(s => s.type === 'ml' || s.type === 'rl' || s.type === 'ensemble');
                
                if (mlStrategies.length === 0) {
                    tableBody.innerHTML = '<tr><td colspan="4" class="text-center">No ML strategies found. Create your first ML strategy!</td></tr>';
                    loadCheckboxes(strategies);
                    return;
                }
                
                mlStrategies.forEach(strategy => {
                    const row = document.createElement('tr');
                    
                    let typeLabel = '';
                    if (strategy.type === 'ml') {
                        typeLabel = '<span class="badge bg-primary">ML</span>';
                    } else if (strategy.type === 'rl') {
                        typeLabel = '<span class="badge bg-success">RL</span>';
                    } else if (strategy.type === 'ensemble') {
                        typeLabel = '<span class="badge bg-info">Ensemble</span>';
                    } else {
                        typeLabel = '<span class="badge bg-secondary">Traditional</span>';
                    }
                    
                    row.innerHTML = `
                        <td>${strategy.name}</td>
                        <td>${typeLabel}</td>
                        <td>${new Date(strategy.created_at).toLocaleString()}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary view-strategy-btn" data-strategy-id="${strategy.id}">View</button>
                            <button class="btn btn-sm btn-outline-success backtest-strategy-btn" data-strategy-id="${strategy.id}">Backtest</button>
                        </td>
                    `;
                    
                    tableBody.appendChild(row);
                });
                
                // Add event listeners to buttons
                document.querySelectorAll('.view-strategy-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const strategyId = this.getAttribute('data-strategy-id');
                        viewStrategy(strategyId);
                    });
                });
                
                document.querySelectorAll('.backtest-strategy-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const strategyId = this.getAttribute('data-strategy-id');
                        backtestStrategy(strategyId);
                    });
                });
                
                // Load checkboxes for ensemble strategy
                loadCheckboxes(strategies);
            })
            .catch(error => {
                console.error('Error loading strategies:', error);
                document.getElementById('strategies-table-body').innerHTML = 
                    `<tr><td colspan="4" class="text-center text-danger">Error loading strategies: ${error}</td></tr>`;
            });
    }
    
    function loadMLModels() {
        fetch('/api/ml/models')
            .then(response => response.json())
            .then(models => {
                const selectElement = document.getElementById('ml-model-select');
                
                // Keep the first option (placeholder)
                const placeholder = selectElement.options[0];
                selectElement.innerHTML = '';
                selectElement.appendChild(placeholder);
                
                // Filter for trained models only
                const trainedModels = models.filter(model => model.is_trained);
                
                if (trainedModels.length === 0) {
                    const option = document.createElement('option');
                    option.disabled = true;
                    option.textContent = 'No trained models available';
                    selectElement.appendChild(option);
                    return;
                }
                
                trainedModels.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = `${model.model_name} (${model.model_type})`;
                    selectElement.appendChild(option);
                });
                
                // Check URL parameters
                const urlParams = new URLSearchParams(window.location.search);
                const modelId = urlParams.get('model');
                
                if (modelId) {
                    selectElement.value = modelId;
                }
            })
            .catch(error => {
                console.error('Error loading ML models:', error);
                showToast('Error loading ML models: ' + error, 'danger');
            });
    }
    
    function loadCheckboxes(strategies) {
        const container = document.getElementById('strategy-checkboxes');
        
        if (strategies.length === 0) {
            container.innerHTML = '<div class="alert alert-warning">No strategies available for ensemble.</div>';
            return;
        }
        
        container.innerHTML = '';
        
        strategies.forEach(strategy => {
            const div = document.createElement('div');
            div.className = 'form-check';
            
            const checkbox = document.createElement('input');
            checkbox.className = 'form-check-input strategy-checkbox';
            checkbox.type = 'checkbox';
            checkbox.id = `strategy-${strategy.id}`;
            checkbox.value = strategy.id;
            
            const label = document.createElement('label');
            label.className = 'form-check-label';
            label.htmlFor = `strategy-${strategy.id}`;
            label.textContent = strategy.name;
            
            div.appendChild(checkbox);
            div.appendChild(label);
            container.appendChild(div);
        });
    }
    
    function createMLStrategy() {
        const form = document.getElementById('createMLStrategyForm');
        
        // Validate form
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        
        // Get form values
        const name = document.getElementById('ml-strategy-name').value;
        const modelId = document.getElementById('ml-model-select').value;
        const positionSizing = parseFloat(document.getElementById('ml-position-sizing').value);
        const useVolatilityScaling = document.getElementById('use-volatility-scaling').checked;
        
        // Create risk control params if using volatility scaling
        let riskControlParams = null;
        
        if (useVolatilityScaling) {
            riskControlParams = {
                volatility_lookback: parseInt(document.getElementById('volatility-lookback').value),
                target_volatility: parseFloat(document.getElementById('target-volatility').value),
                max_leverage: 2.0
            };
        }
        
        // Create request data
        const requestData = {
            name: name,
            model_id: modelId,
            position_sizing: positionSizing,
            risk_control_params: riskControlParams
        };
        
        // Send request to create strategy
        fetch('/api/ml/strategies/ml', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Show success message
                showToast('Strategy created successfully!', 'success');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('createMLStrategyModal'));
                modal.hide();
                
                // Reset form
                form.reset();
                
                // Reload strategies
                loadStrategies();
                
                // View strategy
                viewStrategy(data.strategy_id);
            } else {
                showToast('Error creating strategy: ' + (data.message || 'Unknown error'), 'danger');
            }
        })
        .catch(error => {
            console.error('Error creating strategy:', error);
            showToast('Error creating strategy: ' + error, 'danger');
        });
    }
    
    function createRLStrategy() {
        const form = document.getElementById('createRLStrategyForm');
        
        // Validate form
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        
        // Get form values
        const name = document.getElementById('rl-strategy-name').value;
        const windowSize = parseInt(document.getElementById('window-size').value);
        const actionSpace = parseInt(document.getElementById('action-space').value);
        const positionSizing = parseFloat(document.getElementById('rl-position-sizing').value);
        
        // Create request data
        const requestData = {
            name: name,
            window_size: windowSize,
            action_space: actionSpace,
            position_sizing: positionSizing
        };
        
        // Send request to create strategy
        fetch('/api/ml/strategies/rl', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Show success message
                showToast('Strategy created successfully!', 'success');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('createRLStrategyModal'));
                modal.hide();
                
                // Reset form
                form.reset();
                
                // Reload strategies
                loadStrategies();
                
                // View strategy
                viewStrategy(data.strategy_id);
            } else {
                showToast('Error creating strategy: ' + (data.message || 'Unknown error'), 'danger');
            }
        })
        .catch(error => {
            console.error('Error creating strategy:', error);
            showToast('Error creating strategy: ' + error, 'danger');
        });
    }
    
    function createEnsembleStrategy() {
        const form = document.getElementById('createEnsembleStrategyForm');
        
        // Validate form
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        
        // Get form values
        const name = document.getElementById('ensemble-strategy-name').value;
        const ensembleMethod = document.getElementById('ensemble-method').value;
        const adaptationMethod = document.getElementById('adaptation-method').value;
        
        // Get selected strategies
        const selectedStrategies = [];
        document.querySelectorAll('.strategy-checkbox:checked').forEach(checkbox => {
            selectedStrategies.push(checkbox.value);
        });
        
        if (selectedStrategies.length < 2) {
            showToast('Please select at least two strategies for ensemble', 'warning');
            return;
        }
        
        // Create request data
        const requestData = {
            name: name,
            strategy_ids: selectedStrategies,
            ensemble_method: ensembleMethod,
            adaptation_method: adaptationMethod || null
        };
        
        // Send request to create strategy
        fetch('/api/ml/strategies/ensemble', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Show success message
                showToast('Ensemble strategy created successfully!', 'success');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('createEnsembleStrategyModal'));
                modal.hide();
                
                // Reset form
                form.reset();
                
                // Reload strategies
                loadStrategies();
                
                // View strategy
                viewStrategy(data.strategy_id);
            } else {
                showToast('Error creating ensemble strategy: ' + (data.message || 'Unknown error'), 'danger');
            }
        })
        .catch(error => {
            console.error('Error creating ensemble strategy:', error);
            showToast('Error creating ensemble strategy: ' + error, 'danger');
        });
    }
    
    function viewStrategy(strategyId) {
        fetch(`/api/strategies/${strategyId}`)
            .then(response => response.json())
            .then(strategy => {
                const detailsContainer = document.getElementById('strategy-details-content');
                
                let typeLabel = '';
                if (strategy.type === 'ml') {
                    typeLabel = '<span class="badge bg-primary">ML</span>';
                } else if (strategy.type === 'rl') {
                    typeLabel = '<span class="badge bg-success">RL</span>';
                } else if (strategy.type === 'ensemble') {
                    typeLabel = '<span class="badge bg-info">Ensemble</span>';
                } else {
                    typeLabel = '<span class="badge bg-secondary">Traditional</span>';
                }
                
                detailsContainer.innerHTML = `
                    <div class="mb-3">
                        <h4>${strategy.name} ${typeLabel}</h4>
                        <p><strong>Created:</strong> ${new Date(strategy.created_at).toLocaleString()}</p>
                    </div>
                    
                    <div class="mb-3">
                        <h5>Strategy Details</h5>
                        <pre class="bg-light p-3 rounded">${JSON.stringify(strategy, null, 2)}</pre>
                    </div>
                `;
                
                // Show modal
                const modal = new bootstrap.Modal(document.getElementById('strategyDetailsModal'));
                modal.show();
                
                // Update backtest button
                const backtestBtn = document.getElementById('backtest-strategy-btn');
                backtestBtn.onclick = function() {
                    backtestStrategy(strategyId);
                    modal.hide();
                };
            })
            .catch(error => {
                console.error('Error loading strategy details:', error);
                showToast('Error loading strategy details: ' + error, 'danger');
            });
    }
    
    function backtestStrategy(strategyId) {
        // Redirect to backtest page with strategy ID
        window.location.href = `/backtest?strategy=${strategyId}`;
    }
</script>
{% endblock %}"""
    }
    
    # Create templates
    for filename, content in ml_templates.items():
        template_path = os.path.join(templates_dir, filename)
        
        with open(template_path, 'w') as f:
            f.write(content)
        
        print(f"Created template: {template_path}")


def add_ml_routes():
    """
    Add ML routes to the web interface.
    """
    routes_file = "web/routes.py"
    
    # Check if the routes file exists
    if not os.path.exists(routes_file):
        # Create routes file
        with open(routes_file, 'w') as f:
            f.write("""\"\"\"
Routes for the SCEF web interface.
\"\"\"

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Templates
templates = Jinja2Templates(directory="web/templates")

# Define routes
def register_routes(app):
    @app.get("/ml", response_class=HTMLResponse)
    async def ml_home(request: Request):
        return templates.TemplateResponse("ml/ml_index.html", {"request": request})
    
    @app.get("/ml/models", response_class=HTMLResponse)
    async def ml_models(request: Request):
        return templates.TemplateResponse("ml/ml_models.html", {"request": request})
    
    @app.get("/ml/strategies", response_class=HTMLResponse)
    async def ml_strategies(request: Request):
        return templates.TemplateResponse("ml/ml_strategies.html", {"request": request})
""")
        
        print(f"Created routes file: {routes_file}")
        return True
    
    # Routes file exists, check if ML routes are already added
    with open(routes_file, 'r') as f:
        content = f.read()
    
    if "ml_home" in content:
        print("ML routes are already added to routes.py")
        return True
    
    # Add ML routes
    if "def register_routes(app):" in content:
        # Add routes to existing function
        lines = content.split('\n')
        register_index = -1
        
        for i, line in enumerate(lines):
            if "def register_routes(app):" in line:
                register_index = i
                break
        
        if register_index >= 0:
            # Find the end of the function
            end_index = len(lines)
            for i in range(register_index + 1, len(lines)):
                if lines[i].strip() == "":
                    end_index = i
                    break
            
            # Add ML routes
            ml_routes = [
                "",
                "    @app.get(\"/ml\", response_class=HTMLResponse)",
                "    async def ml_home(request: Request):",
                "        return templates.TemplateResponse(\"ml/ml_index.html\", {\"request\": request})",
                "",
                "    @app.get(\"/ml/models\", response_class=HTMLResponse)",
                "    async def ml_models(request: Request):",
                "        return templates.TemplateResponse(\"ml/ml_models.html\", {\"request\": request})",
                "",
                "    @app.get(\"/ml/strategies\", response_class=HTMLResponse)",
                "    async def ml_strategies(request: Request):",
                "        return templates.TemplateResponse(\"ml/ml_strategies.html\", {\"request\": request})",
                ""
            ]
            
            lines[end_index:end_index] = ml_routes
            
            # Write modified content
            with open(routes_file, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"Added ML routes to {routes_file}")
            return True
        else:
            print(f"Could not find register_routes function in {routes_file}")
            return False
    else:
        # Add register_routes function with ML routes
        ml_routes = """

def register_routes(app):
    @app.get("/ml", response_class=HTMLResponse)
    async def ml_home(request: Request):
        return templates.TemplateResponse("ml/ml_index.html", {"request": request})
    
    @app.get("/ml/models", response_class=HTMLResponse)
    async def ml_models(request: Request):
        return templates.TemplateResponse("ml/ml_models.html", {"request": request})
    
    @app.get("/ml/strategies", response_class=HTMLResponse)
    async def ml_strategies(request: Request):
        return templates.TemplateResponse("ml/ml_strategies.html", {"request": request})
"""
        
        with open(routes_file, 'a') as f:
            f.write(ml_routes)
        
        print(f"Added register_routes function with ML routes to {routes_file}")
        return True


def modify_base_template():
    """
    Modify base.html template to add ML links.
    """
    base_template_path = "web/templates/base.html"
    
    if not os.path.exists(base_template_path):
        print(f"Error: {base_template_path} not found.")
        return False
    
    with open(base_template_path, 'r') as f:
        content = f.read()
    
    # Check if ML links are already added
    if '<a class="nav-link" href="/ml">' in content:
        print("ML links are already added to base.html")
        return True
    
    # Find navbar section
    nav_pattern = '<ul class="navbar-nav'
    if nav_pattern in content:
        nav_index = content.find(nav_pattern)
        if nav_index >= 0:
            # Find the end of the navbar section
            ul_end = content.find('</ul>', nav_index)
            
            if ul_end >= 0:
                # Add ML link
                ml_link = """
                <li class="nav-item">
                    <a class="nav-link" href="/ml">
                        <i class="bi bi-cpu"></i> ML
                    </a>
                </li>"""
                
                # Insert ML link before the end of the navbar
                content = content[:ul_end] + ml_link + content[ul_end:]
                
                # Write modified content
                with open(base_template_path, 'w') as f:
                    f.write(content)
                
                print(f"Added ML link to {base_template_path}")
                return True
    
    print(f"Could not find navbar section in {base_template_path}")
    return False


def update_app_init():
    """
    Update app initialization to include routes.
    """
    app_py_path = "web/app.py"
    
    if not os.path.exists(app_py_path):
        print(f"Error: {app_py_path} not found.")
        return False
    
    with open(app_py_path, 'r') as f:
        content = f.read()
    
    # Check if routes are already included
    if "from web.routes import register_routes" in content:
        print("Routes are already included in app.py")
        return True
    
    # Add routes import
    if "# Create FastAPI app" in content:
        routes_import = """# Import routes
from web.routes import register_routes

# Create FastAPI app"""
        
        content = content.replace("# Create FastAPI app", routes_import)
        
        # Add routes registration
        app_creation_pattern = "app = FastAPI("
        if app_creation_pattern in content:
            # Find the end of the app creation section
            lines = content.split('\n')
            app_line_index = -1
            
            for i, line in enumerate(lines):
                if app_creation_pattern in line:
                    app_line_index = i
                    break
            
            if app_line_index >= 0:
                # Find a good place to insert the routes registration
                # Look for the first line after app creation that's not indented
                insert_index = app_line_index + 1
                
                while insert_index < len(lines) and (
                    lines[insert_index].strip() == '' or 
                    lines[insert_index].startswith(' ') or 
                    lines[insert_index].startswith('\t')
                ):
                    insert_index += 1
                
                # Insert the routes registration
                lines.insert(insert_index, "# Register routes")
                lines.insert(insert_index + 1, "register_routes(app)")
                lines.insert(insert_index + 2, "")
                
                content = '\n'.join(lines)
        
        # Write modified content
        with open(app_py_path, 'w') as f:
            f.write(content)
        
        print(f"Updated {app_py_path} to include routes")
        return True
    
    print(f"Could not find 'Create FastAPI app' section in {app_py_path}")
    return False


def main():
    """
    Main function to integrate ML capabilities into the SCEF web interface.
    """
    print("Integrating ML capabilities into SCEF web interface...")
    
    # Create ML templates
    print("\nCreating ML templates...")
    create_web_ml_templates()
    
    # Add ML routes
    print("\nAdding ML routes...")
    add_ml_routes()
    
    # Modify base.html template
    print("\nModifying base.html template...")
    modify_base_template()
    
    # Update app initialization
    print("\nUpdating app initialization...")
    update_app_init()
    
    # Modify app.py to include ML components
    print("\nModifying app.py to include ML components...")
    modify_app_py()
    
    print("\nIntegration complete!")
    print("\nTo use the ML capabilities:")
    print("1. Restart the SCEF web interface")
    print("2. Navigate to the ML page from the navbar")
    print("3. Create ML models and strategies")
    print("4. Backtest and deploy your ML-based strategies")


if __name__ == "__main__":
    main()