import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DatingMatchmaker:
    def __init__(self, questions_config=None):
        """
        Initialize the matching algorithm with question configurations.
        
        Parameters:
        - questions_config: Dictionary mapping question indices to their weights and match_type
          where match_type can be 'similarity', 'complementary', or 'dealbreaker'
        """
        # Default configuration if none provided
        self.questions_config = questions_config or self._default_question_config()
        
        # Initialize user database
        self.users = {}
        
    def _default_question_config(self):
        """Create default configuration for the 100 questions"""
        config = {}
        
        # Core values questions (high weight, similarity matching)
        for i in range(0, 20):
            config[i] = {'weight': 5.0, 'match_type': 'similarity'}
            
        # Lifestyle preferences (medium weight, similarity matching)
        for i in range(20, 40):
            config[i] = {'weight': 3.0, 'match_type': 'similarity'}
            
        # Personality traits (medium weight, mixed matching)
        for i in range(40, 60):
            config[i] = {'weight': 2.5, 'match_type': 'similarity'}
        
        # Some personality traits benefit from complementary matching
        for i in range(60, 70):
            config[i] = {'weight': 2.0, 'match_type': 'complementary'}
            
        # Interests and hobbies (lower weight, similarity matching)
        for i in range(70, 90):
            config[i] = {'weight': 1.5, 'match_type': 'similarity'}
            
        # Dealbreakers (absolute requirements)
        for i in range(90, 100):
            config[i] = {'weight': 10.0, 'match_type': 'dealbreaker'}
            
        return config
    
    def add_user(self, user_id, answers, preferences=None):
        """
        Add a new user to the database with their question answers
        
        Parameters:
        - user_id: Unique identifier for the user
        - answers: List of 100 numerical answers (normalized between 0-1)
        - preferences: Dictionary of user preferences (age range, location, etc.)
        """
        if len(answers) != 100:
            raise ValueError("Expected 100 answers for all questions")
            
        # Store user data
        self.users[user_id] = {
            'answers': np.array(answers),
            'preferences': preferences or {}
        }
    
    def calculate_compatibility(self, user1_id, user2_id):
        """
        Calculate compatibility score between two users
        
        Returns:
        - score: Compatibility score (0-100%)
        - compatibility_breakdown: Dict with details on match quality
        """
        if user1_id not in self.users or user2_id not in self.users:
            raise ValueError("User not found in database")
            
        user1 = self.users[user1_id]
        user2 = self.users[user2_id]
        
        # Filter by dealbreakers first
        dealbreakers = self._check_dealbreakers(user1, user2)
        if dealbreakers:
            return 0, {'matched': False, 'dealbreakers': dealbreakers}
        
        # Check preference filters (age, location, etc.)
        if not self._check_preferences(user1, user2):
            return 0, {'matched': False, 'reason': 'preference_mismatch'}
        
        # Calculate compatibility scores for different question categories
        similarity_score = self._calculate_similarity(user1, user2)
        complementary_score = self._calculate_complementarity(user1, user2)
        
        # Calculate weighted total compatibility
        total_weight = sum(config['weight'] for config in self.questions_config.values())
        total_score = (
            sum(config['weight'] for i, config in self.questions_config.items() 
                if config['match_type'] == 'similarity') * similarity_score +
            sum(config['weight'] for i, config in self.questions_config.items() 
                if config['match_type'] == 'complementary') * complementary_score
        ) / total_weight
        
        # Convert to percentage
        percentage = round(total_score * 100)
        
        # Create detailed breakdown
        breakdown = {
            'matched': True,
            'overall_score': percentage,
            'similarity_score': round(similarity_score * 100),
            'complementary_score': round(complementary_score * 100),
            'category_scores': self._calculate_category_scores(user1, user2)
        }
        
        return percentage, breakdown
    
    def _check_dealbreakers(self, user1, user2):
        """Check dealbreaker questions to see if users are incompatible"""
        dealbreakers = []
        
        for q_idx, config in self.questions_config.items():
            if config['match_type'] == 'dealbreaker':
                # For dealbreakers, we consider strong disagreement (difference > 0.7) as a deal-breaker
                difference = abs(user1['answers'][q_idx] - user2['answers'][q_idx])
                if difference > 0.7:
                    dealbreakers.append(q_idx)
                    
        return dealbreakers
    
    def _check_preferences(self, user1, user2):
        """Check if users meet each other's basic preference criteria"""
        # This would include filters like age range, location, etc.
        # Simplified implementation for demonstration
        
        if not user1.get('preferences') or not user2.get('preferences'):
            return True  # No preferences set, so all matches are valid
            
        # Example preference check: location
        if 'location' in user1['preferences'] and 'location' in user2['preferences']:
            max_distance = min(
                user1['preferences'].get('max_distance', float('inf')),
                user2['preferences'].get('max_distance', float('inf'))
            )
            
            actual_distance = self._calculate_distance(
                user1['preferences']['location'],
                user2['preferences']['location']
            )
            
            if actual_distance > max_distance:
                return False
                
        # Could add more preference checks here (age, etc.)
        
        return True
    
    def _calculate_distance(self, loc1, loc2):
        """Calculate distance between two locations"""
        # Simplified for demo - would use actual geo-distance calculation
        # Assuming loc1 and loc2 are (latitude, longitude) tuples
        return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5
    
    def _calculate_similarity(self, user1, user2):
        """Calculate similarity score for questions where similarity matters"""
        similarity_indices = [i for i, config in self.questions_config.items() 
                             if config['match_type'] == 'similarity']
        
        if not similarity_indices:
            return 1.0  # No similarity questions
            
        # Extract relevant answers
        u1_answers = user1['answers'][similarity_indices]
        u2_answers = user2['answers'][similarity_indices]
        
        # Calculate weighted similarity
        weights = np.array([self.questions_config[i]['weight'] for i in similarity_indices])
        weights = weights / weights.sum()  # Normalize weights
        
        # 1 - cosine distance = cosine similarity
        similarity = 1 - cosine(u1_answers, u2_answers, w=weights)
        
        return similarity
    
    def _calculate_complementarity(self, user1, user2):
        """Calculate complementary score for questions where differences complement each other"""
        complementary_indices = [i for i, config in self.questions_config.items() 
                               if config['match_type'] == 'complementary']
        
        if not complementary_indices:
            return 1.0  # No complementary questions
            
        # Extract relevant answers
        u1_answers = user1['answers'][complementary_indices]
        u2_answers = user2['answers'][complementary_indices]
        
        # For complementary traits, we want some moderate difference (not too similar, not too different)
        # Ideal difference is around 0.5 on a 0-1 scale
        differences = np.abs(u1_answers - u2_answers)
        
        # Score is highest when difference is close to 0.5, lower when very similar or very different
        comp_scores = 1 - np.abs(differences - 0.5) * 2
        
        # Apply weights
        weights = np.array([self.questions_config[i]['weight'] for i in complementary_indices])
        weights = weights / weights.sum()  # Normalize weights
        
        # Calculate weighted average
        complementary_score = np.average(comp_scores, weights=weights)
        
        return complementary_score
    
    def _calculate_category_scores(self, user1, user2):
        """Calculate scores by question category for detailed breakdown"""
        # Define categories (for demonstration)
        categories = {
            'values': range(0, 20),
            'lifestyle': range(20, 40),
            'personality': range(40, 70),
            'interests': range(70, 90),
        }
        
        category_scores = {}
        
        for category, indices in categories.items():
            # Extract relevant answers
            u1_answers = user1['answers'][list(indices)]
            u2_answers = user2['answers'][list(indices)]
            
            # Calculate similarity for this category
            similarity = 1 - cosine(u1_answers, u2_answers)
            category_scores[category] = round(similarity * 100)
            
        return category_scores
    
    def find_matches(self, user_id, top_n=10):
        """
        Find top N matches for a given user
        
        Parameters:
        - user_id: ID of the user to find matches for
        - top_n: Number of top matches to return
        
        Returns:
        - List of (user_id, compatibility_score, breakdown) tuples
        """
        if user_id not in self.users:
            raise ValueError("User not found in database")
            
        match_scores = []
        
        for potential_match_id in self.users:
            # Skip self-matching
            if potential_match_id == user_id:
                continue
                
            # Calculate compatibility
            score, breakdown = self.calculate_compatibility(user_id, potential_match_id)
            
            # Only include if there's some compatibility
            if score > 0:
                match_scores.append((potential_match_id, score, breakdown))
                
        # Sort by compatibility score (descending)
        match_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return match_scores[:top_n]

    def batch_processing(self):
        """
        Process all users in batches to pre-compute compatibility scores
        Useful for large user bases
        """
        compatibility_matrix = {}
        
        # Process users in pairs
        for user1_id in self.users:
            compatibility_matrix[user1_id] = {}
            
            for user2_id in self.users:
                if user1_id == user2_id:
                    continue
                    
                # Skip if already calculated (symmetric)
                if user2_id in compatibility_matrix and user1_id in compatibility_matrix[user2_id]:
                    compatibility_matrix[user1_id][user2_id] = compatibility_matrix[user2_id][user1_id]
                    continue
                    
                score, breakdown = self.calculate_compatibility(user1_id, user2_id)
                compatibility_matrix[user1_id][user2_id] = (score, breakdown)
                
        return compatibility_matrix


# Example usage:
if __name__ == "__main__":
    # Create the matchmaker with default configuration
    matchmaker = DatingMatchmaker()
    
    # Add some users with random answers (for demonstration)
    import random
    
    # Generate 5 users with random answers
    for user_id in range(1, 6):
        # Generate 100 random answers between 0 and 1
        answers = [random.random() for _ in range(100)]
        
        # Add some preferences
        preferences = {
            'location': (random.uniform(-90, 90), random.uniform(-180, 180)),
            'max_distance': random.uniform(10, 100)
        }
        
        matchmaker.add_user(user_id, answers, preferences)
    
    # Find matches for user 1
    matches = matchmaker.find_matches(1)
    
    # Print top matches
    print(f"Top matches for User 1:")
    for match_id, score, breakdown in matches:
        print(f"User {match_id}: {score}% compatible")
        print(f"  Category scores: {breakdown['category_scores']}")
