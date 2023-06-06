
from .dynamic import DynamicBinaryFocusedTopic, DynamicTopicEmbeddings, DynamicLDA, DynamicBetaFocusedTopic
from .static import DiscreteLatentTopicNVI


class ModelFactory(object):
    _models: dict = {
        'discrete_latent_topic': DiscreteLatentTopicNVI,
        'dynamic_lda': DynamicLDA,
        'neural_dynamical_topic_embeddings': DynamicTopicEmbeddings,
        'dynamic_beta_focused_topic_model': DynamicBetaFocusedTopic,
        'dynamic_binary_focused_topic_model': DynamicBinaryFocusedTopic,
      }

    @classmethod
    def get_instance(cls, model_type: str, **kwargs):
        builder = cls._models.get(model_type)
        if not builder:
            raise ValueError(f'Unknown recognition model {model_type}')
        return builder(**kwargs)
