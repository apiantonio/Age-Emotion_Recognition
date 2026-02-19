import math

class FaceStabilizer:
    def __init__(self, alpha=0.2, distance_threshold=50):
        """
        alpha: Quanto 'pesa' il nuovo valore (0.0 - 1.0).
               0.1 = Molto stabile (lento ad aggiornarsi)
               0.9 = Molto reattivo (più ballerino)
        distance_threshold: Distanza massima in pixel per considerare una faccia "la stessa" del frame prima.
        """
        self.alpha = alpha
        self.distance_threshold = distance_threshold
        # Dizionario per memorizzare lo stato: {id_fittizio: {'center': (x,y), 'age': val}}
        self.tracked_faces = []

    def update(self, coords, raw_ages):
        """
        Prende le coordinate attuali e le età grezze, e restituisce le età stabilizzate.
        """
        current_faces = []
        smoothed_ages = []

        for i, (x, y, w, h) in enumerate(coords):
            center_x = x + w / 2
            center_y = y + h / 2
            current_center = (center_x, center_y)
            raw_age = raw_ages[i]
            
            matched = False
            final_age = raw_age

            # Cerchiamo se questa faccia esisteva nel frame precedente
            for prev_face in self.tracked_faces:
                prev_center = prev_face['center']
                prev_age = prev_face['age']

                # Calcolo distanza Euclidea
                dist = math.hypot(current_center[0] - prev_center[0], current_center[1] - prev_center[1])

                if dist < self.distance_threshold:
                    # È la stessa faccia! Applichiamo lo smoothing
                    # Formula EMA: Valore = alpha * Nuovo + (1-alpha) * Vecchio
                    final_age = self.alpha * raw_age + (1 - self.alpha) * prev_age
                    matched = True
                    break
            
            # Salviamo il nuovo stato
            current_faces.append({'center': current_center, 'age': final_age})
            smoothed_ages.append(final_age)

        # Aggiorniamo la memoria per il prossimo frame
        self.tracked_faces = current_faces
        return smoothed_ages