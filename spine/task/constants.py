

condition_severity_map = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2
}

level_code_map = {
    "l1_l2": 0,
    "l2_l3": 1,
    "l3_l4": 2,
    "l4_l5": 3,
    "l5_s1": 4
}

# spinal_canal, left neural foramen, right neural foramen, left subarticular, right subarticular
conditions_spec_ordered = ['spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1', 'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1']
condition_spec_to_idx = {c: i for i, c in enumerate(conditions_spec_ordered)}
condition_spec_from_idx = {i: c for i, c in enumerate(conditions_spec_ordered)}

conditions_unsided = {
    'neural_foraminal_narrowing',
    'subarticular_stenosis'
    'spinal_canal_stenosis'
}

condition_full_names = [
    'Spinal Canal Stenosis', 
    'Left Neural Foraminal Narrowing', 
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

series = [
    "Axial T2",
    "Sagittal T1",
    "Sagittal T2/STIR"
]

