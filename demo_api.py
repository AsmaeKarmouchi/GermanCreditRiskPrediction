#!/usr/bin/env python3
"""
Script de démonstration et test de l'API
"""
import requests
import json
import time
import sys
from datetime import datetime

class APIDemo:
    """Démonstration de l'API de prédiction"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def wait_for_api(self, max_attempts=30):
        """Attendre que l'API soit disponible"""
        print(" Attente de la disponibilité de l'API...")
        
        for attempt in range(max_attempts):
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print(" API disponible !")
                    return True
            except:
                pass
            
            print(f"   Tentative {attempt + 1}/{max_attempts}...")
            time.sleep(2)
        
        return False
    
    def test_health(self):
        """Test du health check"""
        print("\n Test de santé de l'API")
        print("-" * 40)
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"Status: {response.status_code}")
            print(f"Réponse: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            return response.status_code == 200
        except Exception as e:
            print(f" Erreur: {e}")
            return False
    
    def test_model_info(self):
        """Test des informations du modèle"""
        print("\n Informations du modèle")
        print("-" * 40)
        
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            print(f"Status: {response.status_code}")
            print(f"Réponse: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            return response.status_code == 200
        except Exception as e:
            print(f" Erreur: {e}")
            return False
    
    def test_single_prediction(self):
        """Test de prédiction unique"""
        print("\n Test de prédiction unique")
        print("-" * 40)
        
        # Exemple de client avec profil de risque faible
        client_low_risk = {
            "status": "... >= 200 DM / salary for at least 1 year",
            "duration": 12,
            "credit_history": "existing credits paid back duly till now",
            "purpose": "car (new)",
            "amount": 3000,
            "savings": "... >= 1000 DM", 
            "employment_duration": "... >= 7 years",
            "installment_rate": 2,
            "personal_status_sex": "male : married/widowed",
            "other_debtors": "none",
            "present_residence": 4,
            "property": "real estate",
            "age": 45,
            "other_installment_plans": "none",
            "housing": "own",
            "number_credits": 1,
            "job": "skilled employee/official",
            "people_liable": 1,
            "telephone": "yes",
            "foreign_worker": "yes"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=client_low_risk,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            print(f"Réponse: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200:
                risk_level = result.get('risk_level', 'Unknown')
                confidence = result.get('confidence', 0)
                print(f"\n Résultat: {risk_level} (Confiance: {confidence:.2%})")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f" Erreur: {e}")
            return False
    
    def test_batch_prediction(self):
        """Test de prédiction par lot"""
        print("\n Test de prédiction par lot")
        print("-" * 40)
        
        # Lot de 3 clients avec profils différents
        clients_batch = {
            "data": [
                {
                    "duration": 24,
                    "amount": 5000,
                    "age": 35,
                    "installment_rate": 3
                },
                {
                    "duration": 12,
                    "amount": 2000,
                    "age": 28,
                    "installment_rate": 2
                },
                {
                    "duration": 48,
                    "amount": 15000,
                    "age": 55,
                    "installment_rate": 4
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=clients_batch,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            print(f"Réponse: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200:
                predictions = result.get('predictions', [])
                print(f"\n {len(predictions)} prédictions effectuées:")
                for i, pred in enumerate(predictions, 1):
                    if 'error' not in pred:
                        risk = pred.get('risk_level', 'Unknown')
                        conf = pred.get('confidence', 0)
                        print(f"   Client {i}: {risk} (Confiance: {conf:.2%})")
                    else:
                        print(f"   Client {i}: Erreur - {pred['error']}")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f" Erreur: {e}")
            return False
    
    def test_monitoring_stats(self):
        """Test des statistiques de monitoring"""
        print("\n Statistiques de monitoring")
        print("-" * 40)
        
        try:
            response = self.session.get(f"{self.base_url}/monitoring/stats")
            print(f"Status: {response.status_code}")
            print(f"Réponse: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            return response.status_code == 200
        except Exception as e:
            print(f" Erreur: {e}")
            return False
    
    def run_all_tests(self):
        """Exécuter tous les tests"""
        print("DÉMONSTRATION DE L'API")
        print("=" * 50)
        print(f" Démarré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" URL de base: {self.base_url}")
        
        # Attendre que l'API soit disponible
        if not self.wait_for_api():
            print(" API non disponible. Assurez-vous qu'elle est démarrée avec:")
            print("   python api/prediction_api.py")
            return False
        
        # Tests
        tests = [
            ("Health Check", self.test_health),
            ("Informations Modèle", self.test_model_info),
            ("Prédiction Unique", self.test_single_prediction),
            ("Prédiction par Lot", self.test_batch_prediction),
            ("Statistiques Monitoring", self.test_monitoring_stats)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n Test: {test_name}")
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f" Erreur inattendue: {e}")
                results.append((test_name, False))
        
        # Résumé
        print("\n" + "=" * 50)
        print(" RÉSUMÉ DES TESTS")
        print("=" * 50)
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = " PASSÉ" if success else " ÉCHOUÉ"
            print(f"{test_name}: {status}")
        
        print(f"\n Score: {passed}/{total} tests passés")
        
        if passed == total:
            print(" Tous les tests sont passés ! L'API fonctionne parfaitement.")
        else:
            print(" Certains tests ont échoué. Vérifiez les logs ci-dessus.")
        
        return passed == total

def show_dashboard_info():
    """Afficher les informations sur le dashboard"""
    print("\n" + "=" * 50)
    print(" DASHBOARD STREAMLIT")
    print("=" * 50)
    print("URL: http://localhost:8501")
    print("\n Pages disponibles:")
    print("• Vue d'ensemble - Métriques principales et alertes")
    print("• Performance Modèle - Évaluation détaillée avec graphiques") 
    print("• Monitoring Temps Réel - Suivi des prédictions")
    print("• Analyse des Données - Upload et analyse de fichiers")
    print("• Configuration - Paramètres du système")
    print("\n Pour lancer le dashboard:")
    print("   streamlit run dashboard/ml_dashboard.py")

def main():
    """Fonction principale"""
    if len(sys.argv) > 1 and sys.argv[1] == "--url":
        base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000"
    else:
        base_url = "http://localhost:5000"
    
    # Test de l'API
    demo = APIDemo(base_url)
    api_success = demo.run_all_tests()
    
    # Informations sur le dashboard
    show_dashboard_info()
    
    print("\n" + "=" * 50)
    if api_success:
        print(" DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
    else:
        print("DÉMONSTRATION TERMINÉE AVEC DES ERREURS")
    print("=" * 50)
    
    return api_success

if __name__ == "__main__":
    main()
