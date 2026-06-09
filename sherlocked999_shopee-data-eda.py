import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up the style
plt.style.use('default')
sns.set_palette("husl")


# Load the data
train_df = pd.read_csv("/kaggle/input/shopee-product-matching/train.csv")
print("Dataset shape:", train_df.shape)
train_df.head(10)





# Analyze label distribution
label_count = train_df['label_group'].value_counts()
print("Label Group Statistics:")
print(label_count.describe())

plt.figure(figsize=(15, 10))

# Plot 1: Distribution of label group sizes
plt.subplot(2, 3, 1)
plt.hist(label_count, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Number of Products per Label Group')
plt.ylabel('Frequency')
plt.title('Distribution of Label Group Sizes')
plt.grid(True, alpha=0.3)

# Plot 2: Cumulative distribution
plt.subplot(2, 3, 2)
cumulative = np.cumsum(label_count.values) / np.sum(label_count.values)
plt.plot(range(1, len(cumulative) + 1), cumulative, linewidth=2)
plt.xlabel('Label Group Rank')
plt.ylabel('Cumulative Percentage of Products')
plt.title('Cumulative Distribution of Products\nacross Label Groups')
plt.grid(True, alpha=0.3)

# Plot 3: Top 20 largest label groups
plt.subplot(2, 3, 3)
top_20 = label_count.head(20)
plt.bar(range(len(top_20)), top_20.values, color='lightcoral')
plt.xticks(range(len(top_20)), [f'Group {i+1}' for i in range(len(top_20))], rotation=45)
plt.xlabel('Label Groups')
plt.ylabel('Number of Products')
plt.title('Top 20 Largest Label Groups')

# Plot 4: Box plot of label group sizes
plt.subplot(2, 3, 4)
plt.boxplot(label_count.values, vert=False)
plt.xlabel('Products per Group')
plt.title('Box Plot of Label Group Sizes')

# Plot 5: Log-scale distribution
plt.subplot(2, 3, 5)
plt.hist(label_count, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', log=True)
plt.xlabel('Number of Products per Label Group')
plt.ylabel('Frequency (Log Scale)')
plt.title('Label Group Size Distribution\n(Log Scale)')
plt.grid(True, alpha=0.3)

# Plot 6: Percentage of groups by size
plt.subplot(2, 3, 6)
size_bins = [1, 2, 5, 10, 20, 50, 100, 1000]
size_labels = ['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '100+']
size_counts = []
for i in range(len(size_bins)):
    if i == 0:
        count = (label_count == size_bins[i]).sum()
    elif i == len(size_bins) - 1:
        count = (label_count > size_bins[i-1]).sum()
    else:
        count = ((label_count > size_bins[i-1]) & (label_count <= size_bins[i])).sum()
    size_counts.append(count)

plt.pie(size_counts, labels=size_labels, autopct='%1.1f%%', startangle=90)
plt.title('Label Groups by Size Category')

plt.tight_layout()
plt.show()

# Print detailed statistics
print(f"\nDetailed Label Analysis:")
print(f"Total unique label groups: {len(label_count):,}")
print(f"Groups with only 1 product: {(label_count == 1).sum():,} ({((label_count == 1).sum()/len(label_count)*100):.1f}%)")
print(f"Groups with 2-5 products: {((label_count >= 2) & (label_count <= 5)).sum():,}")
print(f"Groups with 6+ products: {(label_count >= 6).sum():,}")
print(f"Largest group has {label_count.max()} products")
print(f"75% of groups have {label_count.quantile(0.75):.1f} products or less")





# Function to display images for the same product
def display_same_product_images(df, label_group, max_images=8):
    """Display multiple images from the same label group (same product)"""
    product_group = df[df['label_group'] == label_group]
    
    if len(product_group) == 0:
        print(f"No products found for label group {label_group}")
        return
    
    print(f"Label Group {label_group} has {len(product_group)} products")
    
    # Display images
    n_images = min(len(product_group), max_images)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    image_dir = "/kaggle/input/shopee-product-matching/train_images"
    
    for idx, (_, row) in enumerate(product_group.head(n_images).iterrows()):
        try:
            img_path = os.path.join(image_dir, row['image'])
            img = Image.open(img_path)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"Image {idx+1}\nPhash: {row['image_phash'][:8]}...", fontsize=10)
            axes[idx].axis('off')
            
            # Print some info
            if idx == 0:
                print(f"Sample titles in this group:")
            print(f"  {idx+1}. {row['title'][:80]}...")
            
        except Exception as e:
            print(f"Error loading image {row['image']}: {e}")
            axes[idx].text(0.5, 0.5, f"Error\n{row['image']}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, 8):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Analyze some large label groups to see visual variations
large_groups = label_count.head(20).index.tolist()
print("Analyzing images from large label groups (same products):")
print("=" * 60)

for i, group_id in enumerate(large_groups[:20]):  # Show first 3 large groups
    print(f"\nAnalysis for Label Group {group_id}:")
    display_same_product_images(train_df, group_id)





# Find label groups with exactly 2 or 3 products
small_groups_2 = label_count[label_count == 2].index.tolist()
small_groups_3 = label_count[label_count == 3].index.tolist()

print(f"Number of groups with exactly 2 products: {len(small_groups_2)}")
print(f"Number of groups with exactly 3 products: {len(small_groups_3)}")

# Select 10 groups from each category (2 and 3 products)
selected_groups_2 = small_groups_2[:10]
selected_groups_3 = small_groups_3[:10]

# Function to display small product groups
def display_small_product_groups(df, group_ids, group_size):
    """Display image pairs/triplets for small product groups"""
    for i, group_id in enumerate(group_ids):
        product_group = df[df['label_group'] == group_id]
        
        print(f"\n{'='*60}")
        print(f"Group {i+1}: Label Group {group_id} - {len(product_group)} Products")
        print(f"{'='*60}")
        
        # Create subplot based on group size
        fig, axes = plt.subplots(1, group_size, figsize=(4*group_size, 4))
        if group_size == 1:
            axes = [axes]
        
        image_dir = "/kaggle/input/shopee-product-matching/train_images"
        
        for idx, (_, row) in enumerate(product_group.iterrows()):
            try:
                img_path = os.path.join(image_dir, row['image'])
                img = Image.open(img_path)
                
                axes[idx].imshow(img)
                
                # Format title for display
                title = row['title']
                if len(title) > 50:
                    title = title[:47] + "..."
                
                # Split title into multiple lines
                words = title.split()
                lines = []
                current_line = []
                for word in words:
                    if len(' '.join(current_line + [word])) <= 25:
                        current_line.append(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))
                
                display_title = '\n'.join(lines)
                
                axes[idx].set_title(f"Image {idx+1}\nPhash: {row['image_phash'][:6]}...\n{display_title}", 
                                  fontsize=8, pad=10)
                axes[idx].axis('off')
                
                # Print title info
                print(f"  Product {idx+1}: {row['title']}")
                print(f"    Phash: {row['image_phash']}")
                
            except Exception as e:
                print(f"Error loading image {row['image']}: {e}")
                axes[idx].text(0.5, 0.5, f"Error\n{row['image']}", 
                              ha='center', va='center', transform=axes[idx].transAxes, fontsize=8)
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Additional analysis for this group
        unique_phashes = product_group['image_phash'].nunique()
        unique_titles = product_group['title'].nunique()
        print(f"  Analysis: {unique_phashes} unique phashes, {unique_titles} unique titles")
        
        # Check if phashes are identical
        if unique_phashes == 1:
            print("  â†’ Images have IDENTICAL phash (very similar images)")
        else:
            print(f"  â†’ Images have DIFFERENT phashes (visual variations)")
        
        # Check if titles are identical
        if unique_titles == 1:
            print("  â†’ Titles are IDENTICAL")
        else:
            print("  â†’ Titles are DIFFERENT")

# Display groups with 2 products
print("ANALYSIS OF PRODUCTS WITH 2 IMAGES")
print("=" * 60)
display_small_product_groups(train_df, selected_groups_2, 2)

# Display groups with 3 products
print("\n\nANALYSIS OF PRODUCTS WITH 3 IMAGES")
print("=" * 60)
display_small_product_groups(train_df, selected_groups_3, 3)





# Analyze image phash distribution and its relationship with labels
plt.figure(figsize=(15, 5))

# Plot 1: Phash frequency distribution
plt.subplot(1, 3, 1)
phash_count = train_df['image_phash'].value_counts()
plt.hist(phash_count, bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Number of Products per Phash')
plt.ylabel('Frequency')
plt.title('Distribution of Image Phash Frequencies')
plt.grid(True, alpha=0.3)

# Plot 2: Phash consistency with labels
plt.subplot(1, 3, 2)
phash_label_consistency = train_df.groupby('image_phash')['label_group'].nunique()
consistent_phashes = (phash_label_consistency == 1).sum()
inconsistent_phashes = (phash_label_consistency > 1).sum()

plt.pie([consistent_phashes, inconsistent_phashes], 
        labels=['Consistent', 'Inconsistent'], 
        autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Phash Consistency with Label Groups')

# Plot 3: Relationship between phash frequency and label group size
plt.subplot(1, 3, 3)
sample_data = train_df.groupby('label_group').agg({
    'image_phash': lambda x: x.nunique(),
    'posting_id': 'count'
}).sample(1000)  # Sample for better visualization

plt.scatter(sample_data['posting_id'], sample_data['image_phash'], alpha=0.6)
plt.xlabel('Label Group Size (Number of Products)')
plt.ylabel('Number of Unique Phashes')
plt.title('Label Group Size vs Unique Phashes')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Phash Analysis:")
print(f"Unique phashes: {train_df['image_phash'].nunique():,}")
print(f"Phash consistency: {consistent_phashes/(consistent_phashes+inconsistent_phashes)*100:.1f}%")
print(f"Average unique phashes per label group: {train_df.groupby('label_group')['image_phash'].nunique().mean():.2f}")








# Analyze titles and create word clouds
def analyze_titles_and_wordclouds(df, label_group):
    """Analyze titles for a specific label group and create word cloud"""
    product_group = df[df['label_group'] == label_group]
    
    if len(product_group) == 0:
        return
    
    # Combine all titles
    all_titles = ' '.join(product_group['title'].astype(str))
    
    # Create word cloud
    plt.figure(figsize=(15, 6))
    
    # Word cloud
    plt.subplot(1, 2, 1)
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, colormap='viridis').generate(all_titles)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Label Group {label_group}\n({len(product_group)} products)')
    
    # Most common words
    plt.subplot(1, 2, 2)
    words = all_titles.lower().split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(15)
    
    words, counts = zip(*common_words)
    plt.barh(words, counts, color='lightblue')
    plt.xlabel('Frequency')
    plt.title('Top 15 Most Common Words')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    # Print sample titles
    print(f"Sample titles from Label Group {label_group}:")
    for i, title in enumerate(product_group['title'].head(5)):
        print(f"  {i+1}. {title}")

# Analyze text for the same large groups we looked at for images
print("Text Analysis for Large Label Groups:")
print("=" * 50)

for group_id in large_groups[:3]:
    analyze_titles_and_wordclouds(train_df, group_id)





# General text analysis across the entire dataset
plt.figure(figsize=(15, 10))

# Plot 1: Title length distribution
plt.subplot(2, 3, 1)
train_df['title_length'] = train_df['title'].str.len()
plt.hist(train_df['title_length'], bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Title Length (characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Title Lengths')
plt.grid(True, alpha=0.3)

# Plot 2: Title length vs label group size
plt.subplot(2, 3, 2)
label_size_vs_title = train_df.groupby('label_group').agg({'title_length': 'mean', 'posting_id': 'count'})
plt.scatter(label_size_vs_title['posting_id'], label_size_vs_title['title_length'], alpha=0.5)
plt.xlabel('Label Group Size')
plt.ylabel('Average Title Length')
plt.title('Label Group Size vs Average Title Length')
plt.grid(True, alpha=0.3)

# Plot 3: Overall word cloud for entire dataset
plt.subplot(2, 3, 3)
all_titles = ' '.join(train_df['title'].astype(str))
wordcloud = WordCloud(width=600, height=300, background_color='white', 
                     max_words=100).generate(all_titles)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Overall Word Cloud (All Products)')

# Plot 4: Unique titles per label group
plt.subplot(2, 3, 4)
unique_titles_per_group = train_df.groupby('label_group')['title'].nunique()
plt.hist(unique_titles_per_group, bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Number of Unique Titles per Label Group')
plt.ylabel('Frequency')
plt.title('Distribution of Unique Titles per Group')
plt.grid(True, alpha=0.3)

# Plot 5: Title similarity analysis (approximate)
plt.subplot(2, 3, 5)
# Calculate approximate diversity: unique titles / total products per group
title_diversity = unique_titles_per_group / label_count
plt.hist(title_diversity, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Title Diversity Ratio')
plt.ylabel('Frequency')
plt.title('Title Diversity within Label Groups\n(1.0 = all titles unique)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Text Analysis Summary:")
print(f"Average title length: {train_df['title_length'].mean():.1f} characters")
print(f"Shortest title: {train_df['title_length'].min()} characters")
print(f"Longest title: {train_df['title_length'].max()} characters")
print(f"Average unique titles per group: {unique_titles_per_group.mean():.2f}")








def display_joint_perspective(df, label_group, max_display=6):
    """Display images with their titles for the same product group"""
    product_group = df[df['label_group'] == label_group]
    
    if len(product_group) == 0:
        return
    
    n_display = min(len(product_group), max_display)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    image_dir = "/kaggle/input/shopee-product-matching/train_images"
    
    for idx, (_, row) in enumerate(product_group.head(n_display).iterrows()):
        try:
            img_path = os.path.join(image_dir, row['image'])
            img = Image.open(img_path)
            
            axes[idx].imshow(img)
            
            # Create a formatted title (wrap text)
            title = row['title']
            if len(title) > 80:
                title = title[:77] + "..."
            
            # Split title into multiple lines
            words = title.split()
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= 40:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            display_title = '\n'.join(lines)
            
            axes[idx].set_title(f"Phash: {row['image_phash'][:8]}...\n{display_title}", 
                              fontsize=9, pad=10)
            axes[idx].axis('off')
            
        except Exception as e:
            print(f"Error loading image {row['image']}: {e}")
            axes[idx].text(0.5, 0.5, f"Error\n{row['image']}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_display, 6):
        axes[idx].axis('off')
    
    plt.suptitle(f'Joint Perspective: Label Group {label_group} - {len(product_group)} Products\n'
                f'Images with their corresponding titles', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.show()
    
    # Print additional analysis
    print(f"Joint Analysis for Label Group {label_group}:")
    print(f"Number of products: {len(product_group)}")
    print(f"Number of unique image phashes: {product_group['image_phash'].nunique()}")
    print(f"Number of unique titles: {product_group['title'].nunique()}")
    print(f"Title length range: {product_group['title'].str.len().min()} - {product_group['title'].str.len().max()} characters")

# Display joint perspective for different types of groups
print("Joint Perspective Analysis:")
print("=" * 50)

# Analyze different types of groups
small_group = label_count[label_count == 2].index[0]  # A small group
medium_group = label_count[(label_count >= 5) & (label_count <= 10)].index[0]  # Medium group
large_group = large_groups[0]  # Large group

print(f"\n1. Small Group (2 products):")
display_joint_perspective(train_df, small_group)

print(f"\n2. Medium Group (5-10 products):")
display_joint_perspective(train_df, medium_group)

print(f"\n3. Large Group ({label_count[large_group]} products):")
display_joint_perspective(train_df, large_group, max_display=6)





# Cross-modal analysis: Relationship between visual and textual similarity
plt.figure(figsize=(15, 5))

# Plot 1: Phash uniqueness vs title uniqueness
plt.subplot(1, 3, 1)
cross_modal = train_df.groupby('label_group').agg({
    'image_phash': 'nunique',
    'title': 'nunique',
    'posting_id': 'count'
}).sample(1000)  # Sample for clarity

plt.scatter(cross_modal['image_phash'], cross_modal['title'], alpha=0.6)
plt.xlabel('Number of Unique Phashes')
plt.ylabel('Number of Unique Titles')
plt.title('Visual vs Textual Diversity\nwithin Label Groups')
plt.grid(True, alpha=0.3)

# Plot 2: Distribution of phash-to-title ratio
plt.subplot(1, 3, 2)
cross_modal['phash_title_ratio'] = cross_modal['image_phash'] / cross_modal['title']
plt.hist(cross_modal['phash_title_ratio'].replace([np.inf, -np.inf], np.nan).dropna(), 
         bins=30, alpha=0.7, color='teal', edgecolor='black')
plt.xlabel('Phash Diversity / Title Diversity')
plt.ylabel('Frequency')
plt.title('Ratio of Visual to Textual Diversity')
plt.grid(True, alpha=0.3)

# Plot 3: Group size relationship with cross-modal consistency
plt.subplot(1, 3, 3)
plt.scatter(cross_modal['posting_id'], cross_modal['phash_title_ratio'], alpha=0.6)
plt.xlabel('Label Group Size')
plt.ylabel('Phash/Title Diversity Ratio')
plt.title('Group Size vs Cross-modal Diversity')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Cross-modal Analysis Summary:")
print(f"Average unique phashes per group: {cross_modal['image_phash'].mean():.2f}")
print(f"Average unique titles per group: {cross_modal['title'].mean():.2f}")
print(f"Average phash/title diversity ratio: {cross_modal['phash_title_ratio'].median():.2f}")





# Additional analysis: Compare visual and textual similarity patterns in small groups
def analyze_small_group_patterns(df, group_size):
    """Analyze patterns in small groups (2 or 3 products)"""
    small_groups = label_count[label_count == group_size].index
    
    patterns = {
        'same_phash_same_title': 0,
        'same_phash_diff_title': 0,
        'diff_phash_same_title': 0,
        'diff_phash_diff_title': 0
    }
    
    for group_id in small_groups[:100]:  # Sample first 100 groups
        group_data = df[df['label_group'] == group_id]
        unique_phashes = group_data['image_phash'].nunique()
        unique_titles = group_data['title'].nunique()
        
        if unique_phashes == 1 and unique_titles == 1:
            patterns['same_phash_same_title'] += 1
        elif unique_phashes == 1 and unique_titles > 1:
            patterns['same_phash_diff_title'] += 1
        elif unique_phashes > 1 and unique_titles == 1:
            patterns['diff_phash_same_title'] += 1
        else:
            patterns['diff_phash_diff_title'] += 1
    
    return patterns

# Analyze patterns for groups of size 2 and 3
patterns_2 = analyze_small_group_patterns(train_df, 2)
patterns_3 = analyze_small_group_patterns(train_df, 3)

# Visualize the patterns
plt.figure(figsize=(15, 5))

# Groups with 2 products
plt.subplot(1, 2, 1)
labels_2 = ['Same Phash\nSame Title', 'Same Phash\nDiff Title', 'Diff Phash\nSame Title', 'Diff Phash\nDiff Title']
values_2 = list(patterns_2.values())
colors_2 = ['lightgreen', 'lightblue', 'lightcoral', 'lightsalmon']

plt.pie(values_2, labels=labels_2, autopct='%1.1f%%', colors=colors_2, startangle=90)
plt.title(f'Patterns in Groups with 2 Products\n(Total: {sum(values_2)} groups sampled)')

# Groups with 3 products
plt.subplot(1, 2, 2)
labels_3 = ['Same Phash\nSame Title', 'Same Phash\nDiff Title', 'Diff Phash\nSame Title', 'Diff Phash\nDiff Title']
values_3 = list(patterns_3.values())
colors_3 = ['lightgreen', 'lightblue', 'lightcoral', 'lightsalmon']

plt.pie(values_3, labels=labels_3, autopct='%1.1f%%', colors=colors_3, startangle=90)
plt.title(f'Patterns in Groups with 3 Products\n(Total: {sum(values_3)} groups sampled)')

plt.tight_layout()
plt.show()

print("Pattern Analysis Summary for Small Groups:")
print(f"\nGroups with 2 products:")
for pattern, count in patterns_2.items():
    percentage = (count / sum(patterns_2.values())) * 100
    print(f"  {pattern}: {count} ({percentage:.1f}%)")

print(f"\nGroups with 3 products:")
for pattern, count in patterns_3.items():
    percentage = (count / sum(patterns_3.values())) * 100
    print(f"  {pattern}: {count} ({percentage:.1f}%)")





# Show some interesting cases from each pattern
def show_interesting_small_cases(df):
    """Show interesting examples from different pattern categories"""
    
    interesting_cases = {
        'same_phash_same_title': [],
        'same_phash_diff_title': [],
        'diff_phash_same_title': [],
        'diff_phash_diff_title': []
    }
    
    # Find examples for each pattern
    for group_size in [2, 3]:
        groups = label_count[label_count == group_size].index
        
        for group_id in groups:
            group_data = df[df['label_group'] == group_id]
            unique_phashes = group_data['image_phash'].nunique()
            unique_titles = group_data['title'].nunique()
            
            pattern = None
            if unique_phashes == 1 and unique_titles == 1:
                pattern = 'same_phash_same_title'
            elif unique_phashes == 1 and unique_titles > 1:
                pattern = 'same_phash_diff_title'
            elif unique_phashes > 1 and unique_titles == 1:
                pattern = 'diff_phash_same_title'
            else:
                pattern = 'diff_phash_diff_title'
            
            if len(interesting_cases[pattern]) < 2:  # Get 2 examples per pattern
                interesting_cases[pattern].append(group_id)

    image_dir = "/kaggle/input/shopee-product-matching/train_images"
    
    # Display examples
    for pattern, group_ids in interesting_cases.items():
        print(f"\n{'='*70}")
        print(f"PATTERN: {pattern.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        
        for group_id in group_ids:
            group_data = df[df['label_group'] == group_id]
            
            fig, axes = plt.subplots(1, len(group_data), figsize=(4*len(group_data), 4))
            if len(group_data) == 1:
                axes = [axes]
            
            print(f"\nLabel Group {group_id} ({len(group_data)} products):")
            
            for idx, (_, row) in enumerate(group_data.iterrows()):
                try:
                    img_path = os.path.join(image_dir, row['image'])
                    img = Image.open(img_path)
                    
                    axes[idx].imshow(img)
                    
                    # Short title for display
                    title = row['title']
                    if len(title) > 40:
                        title = title[:37] + "..."
                    
                    axes[idx].set_title(f"Product {idx+1}\n{title}", fontsize=9)
                    axes[idx].axis('off')
                    
                    print(f"  Product {idx+1}: {row['title']}")
                    print(f"    Phash: {row['image_phash']}")
                    
                except Exception as e:
                    print(f"Error loading image: {e}")
            
            plt.tight_layout()
            plt.show()

# Display interesting cases
print("INTERESTING CASES FROM DIFFERENT PATTERNS")
show_interesting_small_cases(train_df)











print("=" * 70)
print("COMPREHENSIVE EDA SUMMARY - SHOPEE PRODUCT MATCHING")
print("=" * 70)

print("\nğŸ“Š LABEL ANALYSIS:")
print(f"   â€¢ Total products: {len(train_df):,}")
print(f"   â€¢ Unique label groups: {len(label_count):,}")
print(f"   â€¢ Highly imbalanced: {label_count.max()} products in largest group vs many singles")
print(f"   â€¢ {((label_count == 1).sum()/len(label_count)*100):.1f}% of groups have only 1 product")

print("\nğŸ–¼ï¸� IMAGE PERSPECTIVE:")
print(f"   â€¢ High phash consistency: {consistent_phashes/(consistent_phashes+inconsistent_phashes)*100:.1f}%")
print(f"   â€¢ Same products often have identical or very similar images")
print(f"   â€¢ Phash is a strong signal but not perfect (some inconsistencies exist)")

print("\nğŸ“� TEXT PERSPECTIVE:")
print(f"   â€¢ Average title length: {train_df['title_length'].mean():.1f} characters")
print(f"   â€¢ Significant variation in how same products are described")
print(f"   â€¢ Word clouds show product-specific vocabulary patterns")

print("\nğŸ”— JOINT PERSPECTIVE:")
print(f"   â€¢ Some groups show perfect visual+textual consistency")
print(f"   â€¢ Others show variations in either images or text or both")
print(f"   â€¢ Multi-modal approach is essential for accurate matching")

print("\nğŸ�¯ COMPETITION INSIGHTS:")
print(f"   â€¢ Need to handle class imbalance in training")
print(f"   â€¢ Phash provides strong initial matching signal")
print(f"   â€¢ Text similarity needed for cases with different images")
print(f"   â€¢ Image similarity needed for cases with different descriptions")
print(f"   â€¢ Ensemble of multiple approaches likely to work best")

print("\nğŸ’¡ RECOMMENDATIONS:")
print(f"   1. Use phash for initial candidate generation")
print(f"   2. Implement text embedding models (TF-IDF, BERT, etc.)")
print(f"   3. Use image embedding models (CNN, ResNet, etc.)")
print(f"   4. Combine modalities with careful weighting")
print(f"   5. Handle edge cases where modalities disagree")
print("=" * 70)













