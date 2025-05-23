# Zero-Shot Classification สำหรับ Tech Stack
from transformers import pipeline
import re
from collections import defaultdict

# ===============================
# 1. การประมวลผลหลายรายการ - แยก Tech vs Non-Tech
# ===============================

def batch_classification_with_separation():
    print("📚 การประมวลผลหลายรายการ - แยก Tech vs Non-Tech Jobs")
    print("="*70)
    
    classifier = pipeline("zero-shot-classification", 
                         model="facebook/bart-large-mnli")
    
    # ข้อมูลตัวอย่าง - ผสมระหว่างงาน Tech และ Non-Tech
    job_descriptions = [
        # Tech Jobs
        "Looking for React developer with TypeScript experience and Redux knowledge",
        "Backend position using Django, PostgreSQL, and REST API development", 
        "DevOps engineer familiar with Docker, Kubernetes, and AWS infrastructure",
        "Full-stack developer: Vue.js frontend, Express.js backend, MongoDB database",
        "Mobile developer using React Native, Firebase, and GraphQL",
        "Data scientist with Python, TensorFlow, pandas, and machine learning expertise",
        "Frontend engineer specializing in Angular, RxJS, and responsive design",
        
        # Non-Tech Jobs
        "Marketing manager with social media strategy and campaign management experience",
        "Sales representative for B2B software solutions and client relationship management",
        "HR specialist focusing on recruitment, employee relations, and performance management",
        "Financial analyst with budgeting, forecasting, and Excel modeling skills",
        "Project manager with Agile methodology and team coordination experience",
        "Content writer specializing in technical documentation and blog articles",
        "Customer support representative with problem-solving and communication skills"
    ]
    
    # Categories สำหรับงาน Tech
    tech_categories = [
        "Frontend Development",
        "Backend Development", 
        "Mobile Development",
        "DevOps/Infrastructure",
        "Data Science/ML",
        "Full-Stack Development"
    ]
    
    # Categories สำหรับงานทั่วไป
    general_categories = [
        "Marketing/Sales",
        "Human Resources",
        "Finance/Accounting", 
        "Project Management",
        "Content/Writing",
        "Customer Service",
        "Business Operations"
    ]
    
    # ขั้นตอนที่ 1: แยกว่าเป็นงาน Tech หรือไม่
    tech_vs_general = ["Technology/IT Job", "Non-Technology Job"]
    
    tech_jobs = []
    non_tech_jobs = []
    
    print("🔍 STEP 1: แยกประเภทงาน Tech vs Non-Tech")
    print("-" * 50)
    
    for i, description in enumerate(job_descriptions, 1):
        # จำแนกประเภทหลัก
        main_result = classifier(description, tech_vs_general)
        is_tech = main_result['labels'][0] == "Technology/IT Job"
        confidence = main_result['scores'][0]
        
        job_type = "🖥️  TECH" if is_tech else "👔 NON-TECH"
        print(f"{i:2d}. {job_type} ({confidence:.3f})")
        print(f"    {description[:60]}...")
        
        if is_tech:
            tech_jobs.append((i, description))
        else:
            non_tech_jobs.append((i, description))
        print()
    
    # ขั้นตอนที่ 2: จำแนกงาน Tech แบบละเอียด
    print(f"\n🖥️  TECH JOBS DETAILED CLASSIFICATION ({len(tech_jobs)} jobs)")
    print("=" * 70)
    
    for job_num, description in tech_jobs:
        result = classifier(description, tech_categories)
        
        print(f"\n📋 Job #{job_num}: {description}")
        print("🎯 Tech Categories Ranking:")
        
        for rank, (category, confidence) in enumerate(zip(result['labels'][:3], result['scores'][:3]), 1):
            stars = "⭐" * min(int(confidence * 5), 5)
            print(f"  {rank}. {category}: {confidence:.3f} ({confidence*100:.1f}%) {stars}")
        
        # หา tech keywords
        tech_keywords = extract_tech_keywords(description)
        if tech_keywords:
            print(f"  🔧 Tech Keywords: {', '.join(tech_keywords)}")
    
    # ขั้นตอนที่ 3: จำแนกงาน Non-Tech
    print(f"\n👔 NON-TECH JOBS CLASSIFICATION ({len(non_tech_jobs)} jobs)")
    print("=" * 70)
    
    for job_num, description in non_tech_jobs:
        result = classifier(description, general_categories)
        
        print(f"\n📋 Job #{job_num}: {description}")
        print("🎯 General Categories Ranking:")
        
        for rank, (category, confidence) in enumerate(zip(result['labels'][:3], result['scores'][:3]), 1):
            stars = "⭐" * min(int(confidence * 5), 5)
            print(f"  {rank}. {category}: {confidence:.3f} ({confidence*100:.1f}%) {stars}")

def extract_tech_keywords(text):
    """ดึงคำสำคัญด้านเทคโนโลยี"""
    tech_keywords = {
        # Frontend
        'react', 'vue', 'angular', 'svelte', 'javascript', 'typescript', 'redux', 'rxjs',
        # Backend  
        'django', 'flask', 'fastapi', 'express', 'spring', 'laravel', 'node.js', 'rest api',
        # Database
        'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'graphql',
        # Cloud & DevOps
        'aws', 'gcp', 'azure', 'docker', 'kubernetes', 'firebase',
        # Mobile
        'react native', 'flutter', 'ios', 'android',
        # Data Science
        'python', 'tensorflow', 'pytorch', 'pandas', 'machine learning', 'ml'
    }
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in tech_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword.title())
                
    return found_keywords

# ===============================
# 2. Multi-label Classification พร้อมแสดงคำและ Class
# ===============================

def multi_label_with_word_mapping():
    print("\n🏷️  Multi-label Classification พร้อมการแมปคำกับ Class")
    print("="*70)
    
    classifier = pipeline("zero-shot-classification", 
                         model="facebook/bart-large-mnli")
    
    # Tech Stack categories ที่เฉพาะเจาะจง
    categories = [
        "JavaScript Framework",
        "CSS Framework", 
        "Backend Framework",
        "Database Technology",
        "Cloud Platform",
        "Testing Framework",
        "DevOps Tool",
        "Mobile Technology"
    ]
    
    # ตัวอย่างข้อความที่ซับซ้อน
    complex_texts = [
        "Our web application uses React with Material-UI for frontend, Express.js for backend API, MongoDB for database, Jest for testing, and deployed on AWS EC2",
        
        "Mobile app built with React Native, Firebase for backend services, Redux for state management, and Detox for end-to-end testing",
        
        "Microservices architecture using Spring Boot, PostgreSQL database, Docker containers, Kubernetes orchestration, and Jenkins for CI/CD pipeline",
        
        "E-commerce platform: Vue.js with Vuetify UI, Laravel PHP backend, MySQL database, Redis caching, deployed on Google Cloud Platform"
    ]
    
    # คำสำคัญและ class ที่เกี่ยวข้อง
    keyword_to_class = {
        # JavaScript Frameworks
        'react': 'JavaScript Framework',
        'vue': 'JavaScript Framework', 
        'vue.js': 'JavaScript Framework',
        'angular': 'JavaScript Framework',
        'react native': 'Mobile Technology',
        
        # CSS Frameworks
        'material-ui': 'CSS Framework',
        'bootstrap': 'CSS Framework',
        'tailwind': 'CSS Framework',
        'vuetify': 'CSS Framework',
        
        # Backend Frameworks
        'express': 'Backend Framework',
        'express.js': 'Backend Framework',
        'django': 'Backend Framework',
        'spring boot': 'Backend Framework',
        'laravel': 'Backend Framework',
        
        # Databases
        'mongodb': 'Database Technology',
        'postgresql': 'Database Technology',
        'mysql': 'Database Technology', 
        'redis': 'Database Technology',
        
        # Cloud Platforms
        'aws': 'Cloud Platform',
        'aws ec2': 'Cloud Platform',
        'firebase': 'Cloud Platform',
        'google cloud': 'Cloud Platform',
        'gcp': 'Cloud Platform',
        
        # Testing
        'jest': 'Testing Framework',
        'detox': 'Testing Framework',
        'cypress': 'Testing Framework',
        
        # DevOps
        'docker': 'DevOps Tool',
        'kubernetes': 'DevOps Tool',
        'jenkins': 'DevOps Tool',
        
        # Mobile
        'react native': 'Mobile Technology',
        'flutter': 'Mobile Technology',
        
        # State Management
        'redux': 'JavaScript Framework'
    }
    
    for i, text in enumerate(complex_texts, 1):
        print(f"\n📝 Example {i}:")
        print(f"Text: {text}")
        print("-" * 70)
        
        # Multi-label classification
        result = classifier(text, categories, multi_label=True)
        
        # แสดงผลลัพธ์ที่มี confidence > 0.1
        print("🎯 Multi-label Classification Results:")
        significant_results = []
        
        for label, score in zip(result['labels'], result['scores']):
            if score > 0.1:  # threshold
                significant_results.append((label, score))
                status = "✅" if score > 0.3 else "⚠️" if score > 0.15 else "🔸"
                print(f"  {status} {label}: {score:.3f} ({score*100:.1f}%)")
        
        # แมปคำสำคัญกับ class
        print(f"\n🔍 Keyword → Class Mapping:")
        text_lower = text.lower()
        found_mappings = defaultdict(list)
        
        for keyword, expected_class in keyword_to_class.items():
            if keyword.lower() in text_lower:
                found_mappings[expected_class].append(keyword)
        
        for class_name, keywords in found_mappings.items():
            # หา confidence score ของ class นี้
            class_confidence = next((score for label, score in zip(result['labels'], result['scores']) 
                                   if label == class_name), 0)
            
            keywords_str = ", ".join(keywords)
            print(f"  🔧 {class_name}: [{keywords_str}] → Confidence: {class_confidence:.3f} ({class_confidence*100:.1f}%)")
        
        # เปรียบเทียบระหว่าง predicted vs expected
        print(f"\n📊 Accuracy Analysis:")
        predicted_classes = set(label for label, score in significant_results if score > 0.2)
        expected_classes = set(found_mappings.keys())
        
        correct_predictions = predicted_classes.intersection(expected_classes)
        missed_predictions = expected_classes - predicted_classes
        false_positives = predicted_classes - expected_classes
        
        if correct_predictions:
            print(f"  ✅ Correct: {', '.join(correct_predictions)}")
        if missed_predictions:
            print(f"  ❌ Missed: {', '.join(missed_predictions)}")
        if false_positives:
            print(f"  ⚠️  False Positive: {', '.join(false_positives)}")
        
        accuracy = len(correct_predictions) / len(expected_classes) if expected_classes else 0
        print(f"  📈 Accuracy: {accuracy:.1%}")

# ===============================
# รันตัวอย่าง
# ===============================

if __name__ == "__main__":
    print("🎯 Tech Stack Classification - Advanced Examples")
    print("="*70)
    
    # รันการประมวลผลหลายรายการ
    batch_classification_with_separation()
    
    # รัน Multi-label classification
    multi_label_with_word_mapping()
    
    print(f"\n{'='*70}")
    print("✅ เสร็จสิ้นการทดสอบทั้งหมด!")
    print("\n💡 สรุป Features:")
    print("  🔸 แยกงาน Tech vs Non-Tech อัตโนมัติ")
    print("  🔸 จำแนกประเภทงาน Tech แบบละเอียด") 
    print("  🔸 Multi-label classification พร้อมแมปคำสำคัญ")
    print("  🔸 แสดง confidence score และการวิเคราะห์ความแม่นยำ")
    print("  🔸 ระบุคำสำคัญที่สอดคล้องกับแต่ละ class")