"""
Script to generate HW1_solution.pdf with all questions and answers
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os

# Create PDF
pdf_filename = "HW1_solution.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=18)

# Container for the 'Flowable' objects
elements = []

# Define styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#000000'),
    spaceAfter=30,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading1_style = ParagraphStyle(
    'CustomHeading1',
    parent=styles['Heading1'],
    fontSize=16,
    textColor=colors.HexColor('#1a1a1a'),
    spaceAfter=12,
    spaceBefore=12,
    fontName='Helvetica-Bold'
)

heading2_style = ParagraphStyle(
    'CustomHeading2',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#333333'),
    spaceAfter=10,
    spaceBefore=10,
    fontName='Helvetica-Bold'
)

heading3_style = ParagraphStyle(
    'CustomHeading3',
    parent=styles['Heading3'],
    fontSize=12,
    textColor=colors.HexColor('#444444'),
    spaceAfter=8,
    spaceBefore=8,
    fontName='Helvetica-Bold'
)

normal_style = ParagraphStyle(
    'CustomNormal',
    parent=styles['Normal'],
    fontSize=11,
    textColor=colors.HexColor('#000000'),
    spaceAfter=6,
    alignment=TA_JUSTIFY,
    fontName='Times-Roman'
)

code_style = ParagraphStyle(
    'Code',
    parent=styles['Code'],
    fontSize=9,
    textColor=colors.HexColor('#000000'),
    spaceAfter=6,
    fontName='Courier'
)

# Add title
elements.append(Paragraph("ECE 361E: Edge Computing", title_style))
elements.append(Paragraph("Homework 1 Solutions", title_style))
elements.append(Spacer(1, 0.3*inch))

# ==============================================================================
# PROBLEM 1
# ==============================================================================
elements.append(Paragraph("Problem 1: Logistic Regression for MNIST Classification", heading1_style))
elements.append(Spacer(1, 0.2*inch))

# Problem 1, Question 1
elements.append(Paragraph("Question 1: Train Logistic Regression model and report metrics", heading2_style))
elements.append(Paragraph(
    "Train a logistic regression model for 25 epochs on the normalized MNIST dataset. "
    "Report training accuracy, testing accuracy, total time for training, total time for inference, "
    "average time for inference per image, and GPU memory during training.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Answer:</b>", heading3_style))
elements.append(Paragraph(
    "The logistic regression model was trained for 25 epochs on the normalized MNIST dataset. "
    "The results are shown in Table 1 below:",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

# Table 1 data
table1_data = [
    ['Metric', 'Value'],
    ['Training accuracy [%]', '92.56'],
    ['Testing accuracy [%]', '92.35'],
    ['Total time for training [s]', '199.62'],
    ['Total time for inference [s]', '1.26'],
    ['Average time for inference per image [ms]', '0.1261'],
    ['GPU memory during training [MB]', '17.74']
]

table1 = Table(table1_data, colWidths=[3.5*inch, 1.5*inch])
table1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))

elements.append(table1)
elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph(
    "The model achieves approximately 92% accuracy on both training and test sets, indicating good generalization "
    "without overfitting. The inference time per image is quite fast at approximately 0.13 milliseconds.",
    normal_style
))
elements.append(Spacer(1, 0.2*inch))

# Problem 1, Question 2
elements.append(Paragraph("Question 2: Plot loss and accuracy curves", heading2_style))
elements.append(Paragraph(
    "Plot the loss and accuracy curves as a function of epochs for both training and testing sets.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Answer:</b>", heading3_style))
elements.append(Paragraph(
    "The loss and accuracy curves for the logistic regression model are shown below. "
    "Both plots show steady improvement during training with the model converging around epoch 15-20.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

# Add loss plot
if os.path.exists("p1_q2_loss_plot.png"):
    img_loss = Image("p1_q2_loss_plot.png", width=5*inch, height=3*inch)
    elements.append(img_loss)
    elements.append(Spacer(1, 0.1*inch))

# Add accuracy plot
if os.path.exists("p1_q2_accuracy_plot.png"):
    img_acc = Image("p1_q2_accuracy_plot.png", width=5*inch, height=3*inch)
    elements.append(img_acc)
    elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "The plots show that both training and test losses decrease steadily, while accuracies increase and plateau "
    "around 92%. The close alignment between training and test curves indicates no significant overfitting.",
    normal_style
))

elements.append(PageBreak())

# ==============================================================================
# PROBLEM 2
# ==============================================================================
elements.append(Paragraph("Problem 2: Fully Connected Neural Network for MNIST", heading1_style))
elements.append(Spacer(1, 0.2*inch))

# Problem 2, Question 1
elements.append(Paragraph("Question 1: Does the SimpleFC model overfit?", heading2_style))
elements.append(Paragraph(
    "Train the SimpleFC model (4 fully connected layers) for 25 epochs and analyze whether it overfits.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Answer: Yes, the model overfits.</b>", heading3_style))
elements.append(Paragraph(
    "After viewing the resulting plots of p2_q1 (training/test loss and accuracy per epoch), we can clearly observe overfitting:",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

# Add plots for P2Q1
if os.path.exists("p2_q1_loss_plot.png"):
    img_p2q1_loss = Image("p2_q1_loss_plot.png", width=5*inch, height=3*inch)
    elements.append(img_p2q1_loss)
    elements.append(Spacer(1, 0.1*inch))

if os.path.exists("p2_q1_accuracy_plot.png"):
    img_p2q1_acc = Image("p2_q1_accuracy_plot.png", width=5*inch, height=3*inch)
    elements.append(img_p2q1_acc)
    elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "<b>Evidence of Overfitting:</b>",
    heading3_style
))

elements.append(Paragraph(
    "• <b>Training loss</b> decreases steadily to nearly zero (e.g., from ~0.95 at epoch 1 to ~0.0015 at epoch 25), "
    "while <b>test loss</b> improves at first (down to ~0.07) but then <b>plateaus</b> around 0.07-0.08 and does not "
    "improve (and even increases slightly) as training continues.",
    normal_style
))
elements.append(Spacer(1, 0.05*inch))

elements.append(Paragraph(
    "• <b>Training accuracy</b> rises to almost 100% (e.g., ~99.99% by epoch 25), while <b>test accuracy</b> "
    "improves to about 97-98% and then <b>plateaus</b> there for the rest of training.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "The widening gap between training and test loss (and between very high train accuracy and lower, flat test accuracy) "
    "indicates that the model is memorizing the training set rather than learning features that generalize well. "
    "The SimpleFC network has many parameters (four fully connected layers), so without regularization (e.g., dropout) "
    "it tends to overfit. Adding dropout (as in Problem 2, Question 2) helps reduce this overfitting.",
    normal_style
))

elements.append(PageBreak())

# Problem 2, Question 2
elements.append(Paragraph("Question 2: Dropout experiments — what do you observe?", heading2_style))
elements.append(Paragraph(
    "Train the SimpleFC model with different dropout probabilities (0.0, 0.2, 0.5, 0.8) and analyze the results. "
    "Which dropout probability gives the best and worst results?",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Answer:</b>", heading3_style))
elements.append(Paragraph("<b>What do you observe from the loss plots?</b>", heading3_style))
elements.append(Paragraph(
    "Across the four experiments (dropout 0.0, 0.2, 0.5, 0.8), the loss plots show that as dropout increases, "
    "the <b>gap between training loss and test loss shrinks</b>: training loss no longer drops to nearly zero while "
    "test loss stays high. So dropout reduces overfitting. If dropout is too high, both losses stay higher and the model underfits.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

# Add dropout plots
dropout_values = ['0.0', '0.2', '0.5', '0.8']
for dropout in dropout_values:
    plot_file = f"p2_q2_loss_plot_{dropout}.png"
    if os.path.exists(plot_file):
        elements.append(Paragraph(f"<b>Dropout = {dropout}</b>", heading3_style))
        img = Image(plot_file, width=4.5*inch, height=2.7*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Which dropout probability gives the best results?</b>", heading3_style))
elements.append(Paragraph(
    "The <b>best</b> results (no overfitting, i.e., almost equal train and test loss values) typically come from "
    "<b>dropout 0.2 or 0.5</b>. In those plots, the training and test loss curves are close together and both decrease "
    "to a similar level, so the model generalizes well without memorizing the training set.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Which dropout probability gives the worst results?</b>", heading3_style))
elements.append(Paragraph(
    "The <b>worst</b> results come from two extremes:",
    normal_style
))
elements.append(Spacer(1, 0.05*inch))

elements.append(Paragraph(
    "• <b>Dropout 0.0</b>: Strong overfitting. Training loss drops to nearly zero while test loss plateaus or increases; "
    "the gap between the two curves is large. The model memorizes the training data and does not generalize well.",
    normal_style
))
elements.append(Spacer(1, 0.05*inch))

elements.append(Paragraph(
    "• <b>Dropout 0.8</b>: Underfitting. Both training and test loss stay relatively high because too many neurons are "
    "dropped each epoch; the model cannot learn the task well.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "<b>Explanation:</b> Dropout randomly turns off neurons during training, so the model cannot rely on any single neuron "
    "and is encouraged to learn more robust features. With no dropout (0.0), the model overfits. With moderate dropout "
    "(0.2 or 0.5), train and test loss stay close and performance is best. With very high dropout (0.8), the model is "
    "over-regularized and underfits.",
    normal_style
))

elements.append(PageBreak())

# ==============================================================================
# PROBLEM 3
# ==============================================================================
elements.append(Paragraph("Problem 3: Convolutional Neural Networks for MNIST", heading1_style))
elements.append(Spacer(1, 0.2*inch))

# Problem 3, Question 1
elements.append(Paragraph("Question 1: SimpleCNN Model Analysis", heading2_style))
elements.append(Paragraph(
    "Train the SimpleCNN model (with 2 convolutional layers) for 25 epochs on the normalized MNIST dataset. "
    "Report the model complexity metrics in a table.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Answer:</b>", heading3_style))
elements.append(Paragraph(
    "The SimpleCNN model was trained for 25 epochs. The model complexity metrics are shown in Table 2 below:",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

# Table 2 - SimpleCNN metrics
table2_data = [
    ['Model name', 'MACs', 'FLOPs', '# parameters', 'torchsummary\nsize [KB]', 'Saved model\nsize [KB]'],
    ['SimpleCNN', '3,869,824', '7,739,648', '50,186', '550', '199.60']
]

table2 = Table(table2_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
table2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9)
]))

elements.append(table2)
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "<b>Note:</b> The saved model size is smaller than the torchsummary size because torchsummary estimates the "
    "in-memory size required for all parameters (and sometimes activations), while the saved model only contains "
    "the raw parameter values stored efficiently in a binary file.",
    normal_style
))
elements.append(Spacer(1, 0.2*inch))

# Problem 3, Question 2
elements.append(Paragraph("Question 2: Create your own efficient CNN (MyCNN)", heading2_style))
elements.append(Paragraph(
    "Create your own CNN with at least two convolutional layers and train it for 25 epochs on the normalized MNIST dataset. "
    "Try making this model more efficient (smaller number of parameters and/or smaller model size) compared to SimpleCNN. "
    "Plot the loss and accuracy curves and extend the table with your model's results.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Answer:</b>", heading3_style))
elements.append(Paragraph(
    "A more efficient CNN model (MyCNN) was created by reducing the number of channels in each convolutional layer "
    "(16 and 32 instead of 32 and 64). This resulted in a model with significantly fewer parameters while maintaining "
    "good accuracy. The comparison is shown in Table 3 below:",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

# Table 3 - Comparison
table3_data = [
    ['Model name', 'MACs', 'FLOPs', '# parameters', 'torchsummary\nsize [KB]', 'Saved model\nsize [KB]'],
    ['SimpleCNN', '3,869,824', '7,739,648', '50,186', '550', '199.60'],
    ['MyCNN', '1,031,744', '2,063,488', '20,490', '260', '83.24']
]

table3 = Table(table3_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
table3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9)
]))

elements.append(table3)
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "<b>Efficiency Improvements:</b>",
    heading3_style
))
elements.append(Paragraph(
    "• MyCNN uses approximately <b>59% fewer parameters</b> (20,490 vs 50,186)",
    normal_style
))
elements.append(Paragraph(
    "• MyCNN requires approximately <b>73% fewer MACs</b> (1,031,744 vs 3,869,824)",
    normal_style
))
elements.append(Paragraph(
    "• MyCNN has approximately <b>58% smaller saved model size</b> (83.24 KB vs 199.60 KB)",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "<b>Loss and Accuracy Curves:</b>",
    heading3_style
))
elements.append(Paragraph(
    "The training and test loss/accuracy curves for MyCNN are shown below:",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

# Add MyCNN plots
if os.path.exists("p3_q2_plots.png"):
    img_p3q2 = Image("p3_q2_plots.png", width=6*inch, height=3*inch)
    elements.append(img_p3q2)
    elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "Despite having significantly fewer parameters, MyCNN achieves comparable accuracy to SimpleCNN, demonstrating "
    "that careful architecture design can lead to more efficient models without sacrificing performance.",
    normal_style
))

elements.append(Spacer(1, 0.2*inch))

# Problem 3, Question 3 (Bonus)
elements.append(Paragraph("Question 3 (Bonus): Why is MyCNN better than SimpleCNN?", heading2_style))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Answer:</b>", heading3_style))
elements.append(Paragraph(
    "The model architecture for MyCNN is exactly the same as the SimpleCNN except that we have reduced the number "
    "of channels in each convolutional layer by a factor of 2. This results in a significant reduction in MACs, FLOPs, "
    "number of parameters, and model size, while still maintaining a reasonable accuracy on the MNIST dataset.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "The memory utilized for storing the model is also reduced. The accuracy and loss difference between SimpleCNN "
    "and MyCNN is negligible. This demonstrates that we can design efficient models with fewer resources while still "
    "achieving good performance.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "Since performance between the two models is basically the same and MyCNN is more efficient, <b>MyCNN is a better "
    "choice for deployment in resource-constrained environments</b>. Another reason why MyCNN would be better is because "
    "it decreases the likelihood of overfitting due to having fewer parameters to learn.",
    normal_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph(
    "<b>Key Advantages of MyCNN:</b>",
    heading3_style
))
elements.append(Paragraph(
    "• Reduced computational cost (73% fewer MACs)",
    normal_style
))
elements.append(Paragraph(
    "• Smaller model size (58% reduction) - ideal for edge devices",
    normal_style
))
elements.append(Paragraph(
    "• Lower risk of overfitting due to fewer parameters",
    normal_style
))
elements.append(Paragraph(
    "• Comparable accuracy to SimpleCNN",
    normal_style
))
elements.append(Paragraph(
    "• Better suited for resource-constrained deployment scenarios",
    normal_style
))

# Build PDF
doc.build(elements)
print(f"PDF generated successfully: {pdf_filename}")
