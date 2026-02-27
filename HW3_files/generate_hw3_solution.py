"""
HW3 Solution PDF Generator
Generates a PDF report with all answers, tables, and graphs for ECE 361E HW3
"""

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os

def create_hw3_pdf(output_filename="HW3_Solution.pdf"):
    """Generate the complete HW3 solution PDF"""
    
    # Create PDF document
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=14
    )
    
    # Title
    story.append(Paragraph("ECE 361E: Machine Learning and Data Analytics for Edge AI", title_style))
    story.append(Paragraph("Homework 3 Solution", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Team members and contributions
    story.append(Paragraph("<b>Team Members and Contributions:</b>", heading_style))
    story.append(Paragraph(
        "<b>Ryane Li:</b> Worked on Problem 1, including creating the VGG16 model architecture, "
        "training VGG11, VGG16, and MobileNet-v1 models on Lonestar6, generating the required metrics for "
        "Table 1, and creating the test accuracy comparison plot.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Resul Ovezov:</b> Worked on Problems 2 and 3, including converting models to ONNX format, "
        "deploying VGG11, VGG16, and MobileNet-v1 on RaspberryPi 3B+ and Odroid MC1 edge devices, "
        "collecting inference metrics, measuring power consumption and temperature variations.",
        body_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== PROBLEM 1 ==========
    story.append(Paragraph("Problem 1: PyTorch Evaluation of VGG Models", heading_style))
    
    # Question 2: Table 1
    story.append(Paragraph("Question 2: Training Metrics", subheading_style))
    
    # Read and display Table 1
    df1 = pd.read_csv("table1.csv")
    
    # Create table data with more compact headers
    table1_data = [
        ["Model", "Train\nAcc %", "Test\nAcc %", "Train\nTime (s)", "Params\n(M)", "FLOPs\n(M)", "GPU Mem\n(MB)"]
    ]
    for _, row in df1.iterrows():
        table1_data.append([
            row['Model'],
            f"{row['Training accuracy [%]']:.2f}",
            f"{row['Test accuracy [%]']:.2f}",
            f"{row['Total time for training [s]']:.1f}",
            f"{row['Number of trainable parameters']/1e6:.2f}",
            f"{row['FLOPs [M]']:.1f}",
            f"{row['GPU memory during training [MB]']:.0f}"
        ])
    
    # Create Table 1 with adjusted column widths
    table1 = Table(table1_data, repeatRows=1, colWidths=[0.95*inch, 0.65*inch, 0.65*inch, 0.75*inch, 0.65*inch, 0.65*inch, 0.75*inch])
    table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(Paragraph("<b>Table 1: Training Metrics for VGG11, VGG16, and MobileNet-v1</b>", body_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(table1)
    story.append(Spacer(1, 0.3*inch))
    
    # Question 3: Plot and comparison
    story.append(Paragraph("Question 3: Test Accuracy Comparison", subheading_style))
    
    if os.path.exists("p1_q3_vgg11_vgg16.png"):
        img = Image("p1_q3_vgg11_vgg16.png", width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Analysis and Comparison (VGG11 vs VGG16):</b>", body_style))
    
    story.append(Paragraph(
        "<b>• Accuracy vs. epochs:</b> VGG11 reaches higher accuracy in the very early epochs, but VGG16 "
        "quickly catches up and ultimately achieves better final test accuracy. After 100 epochs, VGG11 ends "
        "at 76.08% test accuracy while VGG16 reaches 78.46%, so VGG16 provides about +2.4 percentage points "
        "better generalization.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>• Training accuracy and overfitting:</b> Both models achieve very high training accuracy "
        "(VGG11: 99.07%, VGG16: 98.35%). The gap between train and test accuracy indicates some overfitting "
        "for both, but the gap is not dramatically worse for VGG16, even though it is deeper.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>• Training time:</b> VGG16 takes longer to train (1622.58 s) than VGG11 (1468.63 s), roughly "
        "10% more wall-clock time for the same number of epochs on the same hardware.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>• Model size and FLOPs:</b> VGG16 is substantially heavier: VGG11 has 9.75M trainable parameters "
        "while VGG16 has 15.25M (about 1.6× more). VGG11 requires 306.6M FLOPs per forward pass, while VGG16 "
        "requires 627.5M FLOPs (about 2× more compute).",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>• GPU memory usage:</b> VGG16 also uses more GPU memory during training (2037 MB) than VGG11 "
        "(939 MB), which matters if GPU memory is a bottleneck.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>Conclusion:</b> VGG16 provides slightly better test accuracy (about 2-3 percentage points) but at "
        "the cost of roughly 2× FLOPs, 1.6× parameters, higher GPU memory usage, and slightly longer training time. "
        "If maximum accuracy is the only goal and compute/memory are plentiful, VGG16 is preferable. However, in "
        "edge- or resource-constrained settings—where training and inference cost matter as much as accuracy—VGG11 "
        "is more attractive because it is significantly cheaper while achieving only a modestly lower test accuracy. "
        "In this homework context, where we care about efficiency on edge devices, we would generally prefer VGG11.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== PROBLEM 2 ==========
    story.append(Paragraph("Problem 2: Deployment on Edge Devices Using ONNX", heading_style))
    
    # Question 2: Table 2
    story.append(Paragraph("Question 2: Inference Metrics on Edge Devices", subheading_style))
    
    # Read and display Table 2
    df2 = pd.read_csv("table2.csv")
    
    # Restructure table2 for better display
    table2_data = [
        ["", "Total Inference Time (s)", "", "RAM Memory (MB)", "", "Accuracy (%)", ""],
        ["Model", "MC1", "RaspberryPi", "MC1", "RaspberryPi", "MC1", "RaspberryPi"]
    ]
    
    for _, row in df2.iterrows():
        table2_data.append([
            row['Model'],
            f"{row['MC1 Total Inference Time (s)']:.2f}",
            f"{row['RaspberryPi Total Inference Time (s)']:.2f}",
            f"{row['MC1 RAM Memory (MB)']:.2f}",
            f"{row['RaspberryPi RAM Memory (MB)']:.2f}",
            f"{row['MC1 Accuracy (%)']:.2f}",
            f"{row['RaspberryPi Accuracy (%)']:.2f}"
        ])
    
    table2 = Table(table2_data, repeatRows=2, colWidths=[1*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.9*inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#5dade2')),
        ('TEXTCOLOR', (0, 0), (-1, 1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 1), 8),
        ('BACKGROUND', (0, 2), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('SPAN', (1, 0), (2, 0)),
        ('SPAN', (3, 0), (4, 0)),
        ('SPAN', (5, 0), (6, 0)),
        ('ROWBACKGROUNDS', (0, 2), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(Paragraph("<b>Table 2: Inference Performance on Edge Devices</b>", body_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(table2)
    story.append(Spacer(1, 0.3*inch))
    
    # Question 3: Power and Temperature plots
    story.append(Paragraph("Question 3: Power Consumption and Temperature Analysis", subheading_style))
    
    if os.path.exists("vgg11_power_consumption_comparison.png"):
        img = Image("vgg11_power_consumption_comparison.png", width=5.5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    
    if os.path.exists("vgg11_temperature_comparison.png"):
        img = Image("vgg11_temperature_comparison.png", width=5.5*inch, height=3.5*inch)
        story.append(Spacer(1, 0.15*inch))
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Read and display Table 3
    df3 = pd.read_csv("table3.csv")
    
    table3_data = [["Model", "MC1 Total Energy (J)", "RaspberryPi Total Energy (J)"]]
    for _, row in df3.iterrows():
        table3_data.append([
            row['Model'],
            f"{row['MC1 Total Energy (J)']:.4f}",
            f"{row['RaspberryPi Total Energy (J)']:.4f}"
        ])
    
    table3 = Table(table3_data, repeatRows=1, colWidths=[1.5*inch, 2*inch, 2*inch])
    table3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(Paragraph("<b>Table 3: Energy Consumption on Edge Devices</b>", body_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(table3)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(PageBreak())
    
    # Performance comparison from notebook
    story.append(Paragraph("<b>Performance Comparison and Analysis:</b>", body_style))
    
    story.append(Paragraph(
        "Based on the VGG11 and VGG16 models for the RaspberryPi and the MC1, the MC1 is best for inference. "
        "This is because while RaspberryPi takes up half the memory space, its inference time and total energy "
        "used is a great amount larger. The total inference time for MC1 was 771.48s while the RaspberryPi took "
        "1289.28s for the VGG11. The large difference between the inference times also resulted in the RaspberryPi "
        "using more energy overall even though the power consumption rate for the MC1 is higher than the Pi.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== PROBLEM 3 ==========
    story.append(Paragraph("Problem 3: MobileNet-v1 on Edge Devices", heading_style))
    
    story.append(Paragraph(
        "Questions 1-2: MobileNet-v1 was trained on CIFAR10 using Lonestar6 and deployed on both edge devices. "
        "The results have been incorporated into the extended versions of Tables 1, 2, and 3 shown above.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Bonus Question 3
    story.append(Paragraph("<b>BONUS Question 3: Comprehensive Model Analysis</b>", subheading_style))
    
    story.append(Paragraph(
        "MobileNet drastically outperforms all the other models on both the MC1 and RaspberryPi.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>For the MC1:</b>",
        body_style
    ))
    
    story.append(Paragraph(
        "MobileNet latency is the fastest at 54.28 ms per image (vs 77.15 ms for VGG11, 111.57 ms for VGG16). "
        "It is the most energy efficient at 0.3600 J per image (vs 0.5912 J for VGG11, 0.9698 J for VGG16). "
        "Accuracy is the highest at 78.49% and has the smallest amount of RAM usage with only 301 MB.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>For the RaspberryPi 3B+:</b>",
        body_style
    ))
    
    story.append(Paragraph(
        "MobileNet latency is the fastest at 101.90 ms per image (vs 128.93 ms for VGG11, 180.16 ms for VGG16). "
        "It is the most energy efficient at 0.6023 J per image (vs 0.7800 J for VGG11, 1.0698 J for VGG16). "
        "Accuracy is the highest at 78.49% and has the smallest amount of RAM usage with only 131 MB.",
        body_style
    ))
    
    # Build PDF
    doc.build(story)
    print(f"PDF successfully generated: {output_filename}")
    return output_filename

if __name__ == "__main__":
    # Change to the directory containing the CSV files and images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Generate the PDF
    output_file = create_hw3_pdf("HW3_Solution.pdf")
    print(f"Solution PDF saved as: {output_file}")
