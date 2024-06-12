from graphviz import Digraph

# Create a Digraph object
dot = Digraph(comment='Xception Model System Architecture')

# Add nodes
dot.node('A', 'Start')
dot.node('B', 'Download dataset\n- Alzheimer’s dataset')
dot.node('C', 'Extract dataset\n- Unzip files\n- List contents')
dot.node('D', 'Data Preparation\n- Load data\n- Resize images\n- Train-test split')
dot.node('E', 'Label Encoding\n- Encode labels')
dot.node('F', 'Data Augmentation\n- ImageDataGenerator setup')
dot.node('G', 'Create and compile model\n- Load Xception\n- Add custom layers\n- Freeze layers')
dot.node('H', 'Set up callbacks\n- ReduceLROnPlateau\n- EarlyStopping\n- ModelCheckpoint')
dot.node('I', 'Train the model\n- Fit model')
dot.node('J', 'Evaluate the model\n- Evaluate on test data\n- Plot results')
dot.node('K', 'Save the model\n- Save to Google Drive')

# Add edges
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK'])

# Render the flowchart
dot.render('xception_model_system_architecture', format='png', view=True)
