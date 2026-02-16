import matplotlib.pyplot as plt

# Set text with larger font size and bold
plt.text(0.5, 0.5, 'Titanic Survival Prediction 🚢', 
         fontsize=40,     # Increased font size
         fontweight='bold', 
         ha='center', 
         va='center')     # Vertical alignment

plt.axis('off')  # Hide axes
plt.show()
