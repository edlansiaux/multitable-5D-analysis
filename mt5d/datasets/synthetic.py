"""
Générateur de données synthétiques multitable
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .base import MT5DDataset, DatasetMetadata, DatasetType

class SyntheticDataGenerator(MT5DDataset):
    """Générateur de données synthétiques multitable"""
    
    def __init__(self, data_dir: str, schema: str = "medical", 
                 scale: str = "medium", download: bool = False):
        super().__init__(data_dir, download)
        self.schema = schema  # "medical", "ecommerce", "financial", "social", "iot"
        self.scale = scale    # "small", "medium", "large", "xlarge"
        self.is_loaded = False
        
        # Définir les tailles selon l'échelle
        self.scale_config = {
            "small": {
                "n_patients": 100, "n_visits": 500, "n_diagnoses": 200,
                "n_customers": 100, "n_transactions": 1000,
                "n_users": 50, "n_products": 100, "n_reviews": 500
            },
            "medium": {
                "n_patients": 1000, "n_visits": 5000, "n_diagnoses": 2000,
                "n_customers": 1000, "n_transactions": 10000,
                "n_users": 500, "n_products": 1000, "n_reviews": 5000
            },
            "large": {
                "n_patients": 10000, "n_visits": 50000, "n_diagnoses": 20000,
                "n_customers": 10000, "n_transactions": 100000,
                "n_users": 5000, "n_products": 10000, "n_reviews": 50000
            },
            "xlarge": {
                "n_patients": 100000, "n_visits": 500000, "n_diagnoses": 200000,
                "n_customers": 100000, "n_transactions": 1000000,
                "n_users": 50000, "n_products": 100000, "n_reviews": 500000
            }
        }
    
    def download(self):
        """Pas de téléchargement pour les données synthétiques"""
        print("Génération de données synthétiques...")
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """Génère les données synthétiques"""
        
        if self.is_loaded:
            return self.tables
        
        print(f"Génération de données synthétiques ({self.schema}, {self.scale})...")
        
        if self.schema == "medical":
            self._generate_medical_data()
        elif self.schema == "ecommerce":
            self._generate_ecommerce_data()
        elif self.schema == "financial":
            self._generate_financial_data()
        elif self.schema == "social":
            self._generate_social_data()
        elif self.schema == "iot":
            self._generate_iot_data()
        else:
            raise ValueError(f"Schéma non supporté: {self.schema}")
        
        self.is_loaded = True
        return self.tables
    
    def _generate_medical_data(self):
        """Génère des données médicales synthétiques"""
        
        config = self.scale_config[self.scale]
        
        # Patients
        n_patients = config["n_patients"]
        patients_df = pd.DataFrame({
            'patient_id': range(n_patients),
            'first_name': [f"Patient_{i}" for i in range(n_patients)],
            'last_name': [f"Last_{i}" for i in range(n_patients)],
            'age': np.random.randint(18, 90, n_patients),
            'gender': np.random.choice(['M', 'F', 'O'], n_patients, p=[0.49, 0.49, 0.02]),
            'blood_type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], n_patients),
            'height_cm': np.random.normal(170, 10, n_patients).astype(int),
            'weight_kg': np.random.normal(70, 15, n_patients).astype(int),
            'smoker': np.random.choice([True, False], n_patients, p=[0.3, 0.7]),
            'registration_date': pd.to_datetime(
                np.random.choice(pd.date_range('2010-01-01', '2023-01-01', freq='D'), 
                               n_patients)
            )
        })
        
        # Diagnoses
        n_diagnoses = config["n_diagnoses"]
        icd_codes = [f'I{i:02d}' for i in range(10, 100, 10)] + \
                   [f'K{i:02d}' for i in range(20, 80, 10)] + \
                   [f'E{i:02d}' for i in range(10, 90, 10)]
        
        diagnoses_df = pd.DataFrame({
            'diagnosis_id': range(n_diagnoses),
            'patient_id': np.random.choice(patients_df['patient_id'], n_diagnoses),
            'icd_code': np.random.choice(icd_codes, n_diagnoses),
            'description': np.random.choice([
                'Hypertension', 'Diabetes Type II', 'Asthma', 'Osteoarthritis',
                'Depression', 'Anxiety Disorder', 'Migraine', 'Hyperlipidemia',
                'COPD', 'Coronary Artery Disease', 'Heart Failure', 'Stroke',
                'Chronic Kidney Disease', 'Cancer', 'Obesity', 'Anemia'
            ], n_diagnoses),
            'diagnosis_date': pd.to_datetime(
                np.random.choice(pd.date_range('2018-01-01', '2023-12-31', freq='D'), 
                               n_diagnoses)
            ),
            'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], n_diagnoses, 
                                        p=[0.6, 0.3, 0.1]),
            'chronic': np.random.choice([True, False], n_diagnoses, p=[0.4, 0.6])
        })
        
        # Visits
        n_visits = config["n_visits"]
        visit_types = ['Routine', 'Emergency', 'Follow-up', 'Specialist']
        
        visits_df = pd.DataFrame({
            'visit_id': range(n_visits),
            'patient_id': np.random.choice(patients_df['patient_id'], n_visits),
            'visit_date': pd.to_datetime(
                np.random.choice(pd.date_range('2020-01-01', '2023-12-31', freq='H'), 
                               n_visits)
            ),
            'visit_type': np.random.choice(visit_types, n_visits),
            'systolic_bp': np.random.normal(120, 20, n_visits).astype(int),
            'diastolic_bp': np.random.normal(80, 15, n_visits).astype(int),
            'heart_rate': np.random.normal(75, 15, n_visits).astype(int),
            'temperature': np.random.normal(36.6, 0.5, n_visits),
            'glucose': np.random.normal(100, 20, n_visits),
            'cholesterol': np.random.normal(200, 40, n_visits)
        })
        
        # Médications
        n_medications = n_patients * 3
        medications_df = pd.DataFrame({
            'prescription_id': range(n_medications),
            'patient_id': np.random.choice(patients_df['patient_id'], n_medications),
            'medication_name': np.random.choice([
                'Aspirin', 'Metformin', 'Lisinopril', 'Atorvastatin',
                'Levothyroxine', 'Metoprolol', 'Amlodipine', 'Omeprazole',
                'Sertraline', 'Simvastatin', 'Losartan', 'Albuterol',
                'Insulin', 'Warfarin', 'Prednisone', 'Gabapentin'
            ], n_medications),
            'dosage_mg': np.random.choice([5, 10, 20, 50, 100, 200, 500], n_medications),
            'frequency': np.random.choice(['QD', 'BID', 'TID', 'QID'], n_medications),
            'start_date': pd.to_datetime(
                np.random.choice(pd.date_range('2020-01-01', '2023-12-31', freq='D'), 
                               n_medications)
            ),
            'end_date': pd.to_datetime(
                np.random.choice(pd.date_range('2021-01-01', '2024-12-31', freq='D'), 
                               n_medications)
            ),
            'refills_left': np.random.randint(0, 5, n_medications)
        })
        
        # Lab Results
        n_labs = n_visits * 2
        labs_df = pd.DataFrame({
            'lab_id': range(n_labs),
            'patient_id': np.random.choice(patients_df['patient_id'], n_labs),
            'visit_id': np.random.choice(visits_df['visit_id'], n_labs),
            'test_name': np.random.choice([
                'CBC', 'Basic Metabolic Panel', 'Lipid Panel', 'Liver Function Tests',
                'Thyroid Panel', 'Hemoglobin A1c', 'Vitamin D', 'Iron Studies'
            ], n_labs),
            'test_date': pd.to_datetime(
                np.random.choice(pd.date_range('2021-01-01', '2023-12-31', freq='H'), 
                               n_labs)
            ),
            'result_value': np.random.normal(0, 1, n_labs),
            'unit': np.random.choice(['mg/dL', 'mmol/L', 'U/L', 'g/dL', '%'], n_labs),
            'reference_range_low': np.random.normal(-2, 0.5, n_labs),
            'reference_range_high': np.random.normal(2, 0.5, n_labs),
            'abnormal_flag': np.random.choice(['Normal', 'High', 'Low'], n_labs, 
                                            p=[0.8, 0.1, 0.1])
        })
        
        self.tables['patients'] = patients_df
        self.tables['diagnoses'] = diagnoses_df
        self.tables['visits'] = visits_df
        self.tables['medications'] = medications_df
        self.tables['lab_results'] = labs_df
        
        print(f"  - Patients: {len(patients_df)}")
        print(f"  - Diagnoses: {len(diagnoses_df)}")
        print(f"  - Visits: {len(visits_df)}")
        print(f"  - Medications: {len(medications_df)}")
        print(f"  - Lab Results: {len(labs_df)}")
    
    def _generate_ecommerce_data(self):
        """Génère des données e-commerce synthétiques"""
        
        config = self.scale_config[self.scale]
        
        # Users
        n_users = config["n_users"]
        users_df = pd.DataFrame({
            'user_id': range(n_users),
            'username': [f"user_{i}" for i in range(n_users)],
            'email': [f"user_{i}@example.com" for i in range(n_users)],
            'age': np.random.randint(18, 70, n_users),
            'gender': np.random.choice(['M', 'F', 'O', None], n_users, p=[0.48, 0.48, 0.02, 0.02]),
            'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP'], n_users),
            'join_date': pd.to_datetime(
                np.random.choice(pd.date_range('2018-01-01', '2023-01-01', freq='D'), 
                               n_users)
            ),
            'total_spent': np.random.exponential(500, n_users).round(2),
            'loyalty_level': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                            n_users, p=[0.5, 0.3, 0.15, 0.05])
        })
        
        # Products
        n_products = config["n_products"]
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 
                     'Sports & Outdoors', 'Beauty', 'Toys', 'Automotive']
        
        products_df = pd.DataFrame({
            'product_id': range(n_products),
            'product_name': [f"Product_{i}" for i in range(n_products)],
            'category': np.random.choice(categories, n_products),
            'subcategory': np.random.choice(['Sub1', 'Sub2', 'Sub3', 'Sub4'], n_products),
            'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD', None], 
                                    n_products, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
            'price': np.random.uniform(5, 500, n_products).round(2),
            'cost': lambda x: x['price'] * np.random.uniform(0.3, 0.7, n_products),
            'stock_quantity': np.random.randint(0, 1000, n_products),
            'avg_rating': np.random.uniform(1, 5, n_products),
            'review_count': np.random.randint(0, 1000, n_products),
            'date_added': pd.to_datetime(
                np.random.choice(pd.date_range('2020-01-01', '2023-12-31', freq='D'), 
                               n_products)
            )
        })
        products_df['cost'] = products_df['price'] * np.random.uniform(0.3, 0.7, n_products)
        
        # Orders
        n_orders = n_users * 10
        orders_df = pd.DataFrame({
            'order_id': range(n_orders),
            'user_id': np.random.choice(users_df['user_id'], n_orders),
            'order_date': pd.to_datetime(
                np.random.choice(pd.date_range('2021-01-01', '2023-12-31', freq='H'), 
                               n_orders)
            ),
            'total_amount': np.random.exponential(100, n_orders).round(2),
            'shipping_cost': np.random.uniform(0, 20, n_orders).round(2),
            'tax_amount': lambda x: x['total_amount'] * np.random.uniform(0.05, 0.15, n_orders),
            'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Apple Pay', 'Google Pay'], 
                                             n_orders),
            'shipping_status': np.random.choice(['Pending', 'Shipped', 'Delivered', 'Cancelled'], 
                                              n_orders, p=[0.1, 0.3, 0.55, 0.05]),
            'delivery_date': lambda x: x['order_date'] + pd.to_timedelta(
                np.random.randint(1, 14, n_orders), unit='D'
            )
        })
        orders_df['tax_amount'] = orders_df['total_amount'] * np.random.uniform(0.05, 0.15, n_orders)
        orders_df['delivery_date'] = orders_df['order_date'] + pd.to_timedelta(
            np.random.randint(1, 14, n_orders), unit='D'
        )
        
        # Order Items
        n_order_items = n_orders * 3
        order_items_df = pd.DataFrame({
            'order_item_id': range(n_order_items),
            'order_id': np.random.choice(orders_df['order_id'], n_order_items),
            'product_id': np.random.choice(products_df['product_id'], n_order_items),
            'quantity': np.random.randint(1, 5, n_order_items),
            'unit_price': np.random.uniform(5, 200, n_order_items).round(2),
            'discount': np.random.uniform(0, 0.5, n_order_items).round(2)
        })
        
        # Reviews
        n_reviews = config["n_reviews"]
        reviews_df = pd.DataFrame({
            'review_id': range(n_reviews),
            'user_id': np.random.choice(users_df['user_id'], n_reviews),
            'product_id': np.random.choice(products_df['product_id'], n_reviews),
            'order_id': np.random.choice(orders_df['order_id'], n_reviews),
            'rating': np.random.randint(1, 6, n_reviews),
            'review_text': [f"Review text for product {i}" for i in range(n_reviews)],
            'review_date': pd.to_datetime(
                np.random.choice(pd.date_range('2022-01-01', '2023-12-31', freq='D'), 
                               n_reviews)
            ),
            'helpful_votes': np.random.randint(0, 50, n_reviews),
            'verified_purchase': np.random.choice([True, False], n_reviews, p=[0.7, 0.3])
        })
        
        self.tables['users'] = users_df
        self.tables['products'] = products_df
        self.tables['orders'] = orders_df
        self.tables['order_items'] = order_items_df
        self.tables['reviews'] = reviews_df
        
        print(f"  - Users: {len(users_df)}")
        print(f"  - Products: {len(products_df)}")
        print(f"  - Orders: {len(orders_df)}")
        print(f"  - Order Items: {len(order_items_df)}")
        print(f"  - Reviews: {len(reviews_df)}")
    
    def _generate_financial_data(self):
        """Génère des données financières synthétiques"""
        # Utiliser le FinancialTransactionsLoader
        from .financial import FinancialTransactionsLoader
        
        financial_loader = FinancialTransactionsLoader(
            data_dir=str(self.data_dir),
            dataset="synthetic",
            include_fraud=True
        )
        
        self.tables = financial_loader.load()
    
    def _generate_social_data(self):
        """Génère des données de réseau social synthétiques"""
        
        config = self.scale_config[self.scale]
        
        # Users
        n_users = config["n_users"]
        users_df = pd.DataFrame({
            'user_id': range(n_users),
            'username': [f"user_{i}" for i in range(n_users)],
            'display_name': [f"User {i}" for i in range(n_users)],
            'age': np.random.randint(13, 80, n_users),
            'location': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 
                                        'Sydney', 'Berlin', 'Mumbai'], n_users),
            'join_date': pd.to_datetime(
                np.random.choice(pd.date_range('2010-01-01', '2023-01-01', freq='D'), 
                               n_users)
            ),
            'verified': np.random.choice([True, False], n_users, p=[0.1, 0.9]),
            'followers_count': np.random.poisson(1000, n_users),
            'following_count': np.random.poisson(500, n_users),
            'post_count': np.random.poisson(200, n_users)
        })
        
        # Posts
        n_posts = n_users * 20
        posts_df = pd.DataFrame({
            'post_id': range(n_posts),
            'user_id': np.random.choice(users_df['user_id'], n_posts),
            'content': [f"Post content {i}" for i in range(n_posts)],
            'post_date': pd.to_datetime(
                np.random.choice(pd.date_range('2020-01-01', '2023-12-31', freq='H'), 
                               n_posts)
            ),
            'likes_count': np.random.poisson(100, n_posts),
            'comments_count': np.random.poisson(20, n_posts),
            'shares_count': np.random.poisson(10, n_posts),
            'post_type': np.random.choice(['text', 'image', 'video', 'link'], 
                                        n_posts, p=[0.4, 0.3, 0.2, 0.1]),
            'language': np.random.choice(['en', 'es', 'fr', 'de', 'ja'], n_posts)
        })
        
        # Comments
        n_comments = n_posts * 5
        comments_df = pd.DataFrame({
            'comment_id': range(n_comments),
            'post_id': np.random.choice(posts_df['post_id'], n_comments),
            'user_id': np.random.choice(users_df['user_id'], n_comments),
            'comment_text': [f"Comment {i}" for i in range(n_comments)],
            'comment_date': pd.to_datetime(
                np.random.choice(pd.date_range('2020-01-01', '2023-12-31', freq='H'), 
                               n_comments)
            ),
            'likes_count': np.random.poisson(10, n_comments)
        })
        
        # Follows (relations entre utilisateurs)
        n_follows = n_users * 10
        follows_df = pd.DataFrame({
            'follow_id': range(n_follows),
            'follower_id': np.random.choice(users_df['user_id'], n_follows),
            'followed_id': np.random.choice(users_df['user_id'], n_follows),
            'follow_date': pd.to_datetime(
                np.random.choice(pd.date_range('2018-01-01', '2023-12-31', freq='D'), 
                               n_follows)
            ),
            'notifications_enabled': np.random.choice([True, False], n_follows, p=[0.7, 0.3])
        })
        
        # Messages
        n_messages = n_users * 50
        messages_df = pd.DataFrame({
            'message_id': range(n_messages),
            'sender_id': np.random.choice(users_df['user_id'], n_messages),
            'receiver_id': np.random.choice(users_df['user_id'], n_messages),
            'message_text': [f"Message {i}" for i in range(n_messages)],
            'send_date': pd.to_datetime(
                np.random.choice(pd.date_range('2021-01-01', '2023-12-31', freq='H'), 
                               n_messages)
            ),
            'read': np.random.choice([True, False], n_messages, p=[0.8, 0.2]),
            'read_date': lambda x: x['send_date'] + pd.to_timedelta(
                np.random.randint(0, 24*60, n_messages), unit='m'
            )
        })
        messages_df.loc[~messages_df['read'], 'read_date'] = pd.NaT
        
        self.tables['users'] = users_df
        self.tables['posts'] = posts_df
        self.tables['comments'] = comments_df
        self.tables['follows'] = follows_df
        self.tables['messages'] = messages_df
        
        print(f"  - Users: {len(users_df)}")
        print(f"  - Posts: {len(posts_df)}")
        print(f"  - Comments: {len(comments_df)}")
        print(f"  - Follows: {len(follows_df)}")
        print(f"  - Messages: {len(messages_df)}")
    
    def _generate_iot_data(self):
        """Génère des données IoT synthétiques"""
        
        config = self.scale_config[self.scale]
        
        # Devices
        n_devices = 100
        devices_df = pd.DataFrame({
            'device_id': range(n_devices),
            'device_type': np.random.choice(['sensor', 'actuator', 'gateway', 'controller'], 
                                          n_devices, p=[0.5, 0.2, 0.2, 0.1]),
            'location': np.random.choice(['Living Room', 'Kitchen', 'Bedroom', 'Office', 
                                        'Garage', 'Garden'], n_devices),
            'manufacturer': np.random.choice(['CompanyA', 'CompanyB', 'CompanyC'], n_devices),
            'model': np.random.choice(['ModelX', 'ModelY', 'ModelZ'], n_devices),
            'firmware_version': np.random.choice(['1.0', '1.1', '2.0', '2.1'], n_devices),
            'installation_date': pd.to_datetime(
                np.random.choice(pd.date_range('2020-01-01', '2023-01-01', freq='D'), 
                               n_devices)
            ),
            'battery_level': np.random.uniform(0, 100, n_devices),
            'status': np.random.choice(['online', 'offline', 'error'], 
                                     n_devices, p=[0.85, 0.1, 0.05])
        })
        
        # Sensor Readings
        n_readings = 10000
        sensor_types = ['temperature', 'humidity', 'pressure', 'light', 'motion', 'sound']
        
        readings_df = pd.DataFrame({
            'reading_id': range(n_readings),
            'device_id': np.random.choice(devices_df['device_id'], n_readings),
            'timestamp': pd.to_datetime(
                np.random.choice(pd.date_range('2023-01-01', '2023-12-31', freq='min'), 
                               n_readings)
            ),
            'sensor_type': np.random.choice(sensor_types, n_readings),
            'value': np.random.normal(0, 1, n_readings),
            'unit': np.random.choice(['°C', '%', 'hPa', 'lux', 'boolean', 'dB'], n_readings),
            'battery_used': np.random.exponential(0.01, n_readings)
        })
        
        # Events
        n_events = 500
        events_df = pd.DataFrame({
            'event_id': range(n_events),
            'device_id': np.random.choice(devices_df['device_id'], n_events),
            'event_type': np.random.choice(['motion_detected', 'temperature_alert', 
                                          'device_offline', 'battery_low', 'error'], 
                                         n_events),
            'event_time': pd.to_datetime(
                np.random.choice(pd.date_range('2023-06-01', '2023-12-31', freq='H'), 
                               n_events)
            ),
            'severity': np.random.choice(['low', 'medium', 'high', 'critical'], 
                                       n_events, p=[0.5, 0.3, 0.15, 0.05]),
            'description': [f"Event {i}" for i in range(n_events)],
            'acknowledged': np.random.choice([True, False], n_events, p=[0.7, 0.3]),
            'resolved': np.random.choice([True, False], n_events, p=[0.6, 0.4])
        })
        
        # Commands
        n_commands = 200
        commands_df = pd.DataFrame({
            'command_id': range(n_commands),
            'device_id': np.random.choice(devices_df['device_id'], n_commands),
            'command_type': np.random.choice(['turn_on', 'turn_off', 'set_value', 
                                            'reboot', 'update'], n_commands),
            'command_time': pd.to_datetime(
                np.random.choice(pd.date_range('2023-06-01', '2023-12-31', freq='H'), 
                               n_commands)
            ),
            'parameters': [f"{{'value': {np.random.randn()}}}" for _ in range(n_commands)],
            'status': np.random.choice(['pending', 'executed', 'failed'], 
                                     n_commands, p=[0.2, 0.75, 0.05]),
            'execution_time': lambda x: x['command_time'] + pd.to_timedelta(
                np.random.randint(0, 60, n_commands), unit='s'
            )
        })
        commands_df['execution_time'] = commands_df['command_time'] + pd.to_timedelta(
            np.random.randint(0, 60, n_commands), unit='s'
        )
        
        # Network
        n_connections = 300
        connections_df = pd.DataFrame({
            'connection_id': range(n_connections),
            'source_device': np.random.choice(devices_df['device_id'], n_connections),
            'target_device': np.random.choice(devices_df['device_id'], n_connections),
            'connection_type': np.random.choice(['wifi', 'bluetooth', 'zigbee', 'ethernet'], 
                                              n_connections),
            'signal_strength': np.random.uniform(-100, -30, n_connections),
            'last_seen': pd.to_datetime(
                np.random.choice(pd.date_range('2023-11-01', '2023-12-31', freq='H'), 
                               n_connections)
            )
        })
        
        self.tables['devices'] = devices_df
        self.tables['sensor_readings'] = readings_df
        self.tables['events'] = events_df
        self.tables['commands'] = commands_df
        self.tables['connections'] = connections_df
        
        print(f"  - Devices: {len(devices_df)}")
        print(f"  - Sensor Readings: {len(readings_df)}")
        print(f"  - Events: {len(events_df)}")
        print(f"  - Commands: {len(commands_df)}")
        print(f"  - Connections: {len(connections_df)}")
    
    def get_relationships(self) -> List[Tuple]:
        """Retourne les relations pour le schéma actuel"""
        
        relationships = []
        
        if self.schema == "medical":
            if 'patients' in self.tables and 'diagnoses' in self.tables:
                relationships.append(
                    ('diagnoses', 'patient_id', 'patients', 'patient_id', 'patient_diagnosis')
                )
            
            if 'patients' in self.tables and 'visits' in self.tables:
                relationships.append(
                    ('visits', 'patient_id', 'patients', 'patient_id', 'patient_visit')
                )
            
            if 'patients' in self.tables and 'medications' in self.tables:
                relationships.append(
                    ('medications', 'patient_id', 'patients', 'patient_id', 'patient_medication')
                )
            
            if 'visits' in self.tables and 'lab_results' in self.tables:
                relationships.append(
                    ('lab_results', 'visit_id', 'visits', 'visit_id', 'visit_lab')
                )
        
        elif self.schema == "ecommerce":
            if 'users' in self.tables and 'orders' in self.tables:
                relationships.append(
                    ('orders', 'user_id', 'users', 'user_id', 'user_order')
                )
            
            if 'orders' in self.tables and 'order_items' in self.tables:
                relationships.append(
                    ('order_items', 'order_id', 'orders', 'order_id', 'order_item')
                )
            
            if 'products' in self.tables and 'order_items' in self.tables:
                relationships.append(
                    ('order_items', 'product_id', 'products', 'product_id', 'product_order')
                )
            
            if 'users' in self.tables and 'reviews' in self.tables:
                relationships.append(
                    ('reviews', 'user_id', 'users', 'user_id', 'user_review')
                )
            
            if 'products' in self.tables and 'reviews' in self.tables:
                relationships.append(
                    ('reviews', 'product_id', 'products', 'product_id', 'product_review')
                )
        
        elif self.schema == "social":
            if 'users' in self.tables and 'posts' in self.tables:
                relationships.append(
                    ('posts', 'user_id', 'users', 'user_id', 'user_post')
                )
            
            if 'posts' in self.tables and 'comments' in self.tables:
                relationships.append(
                    ('comments', 'post_id', 'posts', 'post_id', 'post_comment')
                )
            
            if 'users' in self.tables and 'comments' in self.tables:
                relationships.append(
                    ('comments', 'user_id', 'users', 'user_id', 'user_comment')
                )
            
            if 'users' in self.tables and 'follows' in self.tables:
                relationships.append(
                    ('follows', 'follower_id', 'users', 'user_id', 'user_follows')
                )
            
            if 'users' in self.tables and 'messages' in self.tables:
                relationships.append(
                    ('messages', 'sender_id', 'users', 'user_id', 'user_message_sent')
                )
                relationships.append(
                    ('messages', 'receiver_id', 'users', 'user_id', 'user_message_received')
                )
        
        elif self.schema == "iot":
            if 'devices' in self.tables and 'sensor_readings' in self.tables:
                relationships.append(
                    ('sensor_readings', 'device_id', 'devices', 'device_id', 'device_reading')
                )
            
            if 'devices' in self.tables and 'events' in self.tables:
                relationships.append(
                    ('events', 'device_id', 'devices', 'device_id', 'device_event')
                )
            
            if 'devices' in self.tables and 'commands' in self.tables:
                relationships.append(
                    ('commands', 'device_id', 'devices', 'device_id', 'device_command')
                )
            
            if 'devices' in self.tables and 'connections' in self.tables:
                relationships.append(
                    ('connections', 'source_device', 'devices', 'device_id', 'device_connection_source')
                )
                relationships.append(
                    ('connections', 'target_device', 'devices', 'device_id', 'device_connection_target')
                )
        
        return relationships
    
    def get_metadata(self) -> DatasetMetadata:
        """Retourne les métadonnées des données synthétiques"""
        
        if not self.metadata:
            if not self.tables:
                self.load()
            
            schema_names = {
                "medical": "Medical Synthetic Data",
                "ecommerce": "E-commerce Synthetic Data",
                "financial": "Financial Synthetic Data",
                "social": "Social Network Synthetic Data",
                "iot": "IoT Synthetic Data"
            }
            
            schema_descriptions = {
                "medical": "Synthetic medical dataset with patients, diagnoses, visits, medications, and lab results",
                "ecommerce": "Synthetic e-commerce dataset with users, products, orders, and reviews",
                "financial": "Synthetic financial transactions dataset",
                "social": "Synthetic social network dataset with users, posts, comments, follows, and messages",
                "iot": "Synthetic IoT dataset with devices, sensor readings, events, and commands"
            }
            
            self.metadata = DatasetMetadata(
                name=schema_names.get(self.schema, "Synthetic Data"),
                type=DatasetType.SYNTHETIC,
                description=schema_descriptions.get(self.schema, "Synthetic dataset"),
                source="MT5D Synthetic Data Generator",
                license="MIT",
                num_tables=len(self.tables),
                total_rows=sum(len(df) for df in self.tables.values()),
                total_columns=sum(len(df.columns) for df in self.tables.values()),
                has_temporal_data=True,
                has_relationships=len(self.get_relationships()) > 0,
                version="1.0"
            )
        
        return self.metadata

# Enregistrer dans le registre
from .base import DatasetRegistry
DatasetRegistry.register("synthetic", SyntheticDataGenerator)
DatasetRegistry.register("synthetic-medical", lambda data_dir, **kwargs: 
                        SyntheticDataGenerator(data_dir, schema="medical", **kwargs))
DatasetRegistry.register("synthetic-ecommerce", lambda data_dir, **kwargs: 
                        SyntheticDataGenerator(data_dir, schema="ecommerce", **kwargs))
DatasetRegistry.register("synthetic-social", lambda data_dir, **kwargs: 
                        SyntheticDataGenerator(data_dir, schema="social", **kwargs))
DatasetRegistry.register("synthetic-iot", lambda data_dir, **kwargs: 
                        SyntheticDataGenerator(data_dir, schema="iot", **kwargs))
