import cv2
import numpy as np
from PIL import Image

from libs.interfaces import Character, Node
from libs.solver.spectral_solver import SMSolver as Solver, SimilarityMeasure
from utils.component_utils import transfer_color
from utils.image_utils import imshow


def run_with_node_matching():
    reference_sketch_path = "./data/suneo_image/reference_sketch.png"
    reference_color_path = "./data/suneo_image/reference_color.png"
    target_sketch_path = "./data/suneo_image/target_sketch.png"

    # reading
    image_reference = cv2.imread(reference_sketch_path, 0)
    image_reference_color = cv2.imread(reference_color_path)
    image_reference_color = cv2.cvtColor(image_reference_color, cv2.COLOR_BGR2RGB)

    image_target = cv2.imread(target_sketch_path, 0)

    # solving
    character_reference = Character(image_reference)
    character_target = Character(image_target)
    n_reference = len(character_reference)
    n_target = len(character_target)

    similarity_matrix = np.zeros(shape=(n_reference, n_target), dtype=np.float32)
    for i_reference in range(n_reference):
        for i_target in range(n_target):
            sim = SimilarityMeasure.calculate_for_node(character_reference[i_reference], character_target[i_target],
                                                       sigma1=1.5)

            similarity_matrix[i_reference, i_target] = sim

    corresponding_reference_ids = np.argmax(similarity_matrix, axis=0)
    matching = [(reference_id, target_id) for target_id, reference_id in enumerate(corresponding_reference_ids)]

    # visualize
    visualize_image = transfer_color(image_reference_color, image_target,
                                     character_reference, character_target, matching)
    Image.fromarray(visualize_image).save('result_of_node_matching.png')


def run_with_spectral_matching():
    reference_sketch_path = "./data/suneo_image/reference_sketch.png"
    reference_color_path = "./data/suneo_image/reference_color.png"
    target_sketch_path = "./data/suneo_image/target_sketch.png"

    # reading
    image_reference = cv2.imread(reference_sketch_path, 0)
    image_reference_color = cv2.imread(reference_color_path)
    image_reference_color = cv2.cvtColor(image_reference_color, cv2.COLOR_BGR2RGB)

    image_target = cv2.imread(target_sketch_path, 0)

    # solving
    character_reference = Character(image_reference)
    character_target = Character(image_target)
    n_reference = len(character_reference)
    n_target = len(character_target)

    solver = Solver(reference=character_reference, target=character_target)
    M, candidates = solver.setup_affinity_mat()
    matching = solver.solve_using_spectral_matching(M, candidates, n_reference, n_target)

    # visualize
    visualize_image = transfer_color(image_reference_color, image_target,
                                     character_reference, character_target, matching)
    Image.fromarray(visualize_image).save('result_of_spectral_matching.png')


if __name__ == '__main__':
    run_with_spectral_matching()
    run_with_node_matching()
