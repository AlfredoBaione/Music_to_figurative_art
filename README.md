# Music_to_figurative_art
This repository provides the codes for generating artistic images from music inputs.
It was adopted to produce the paper "A Multimodal Visual Language Model for Musical Encoding", whose abstract is the following:

"In the era of deep generative modeling, generative art has become one of the most challenging research fields. Understanding the potential of machine learning within the artistic domain is key to exploring AI’s role in human-machine co-creative processes. In this context, the paper focuses on developing a multimodal deep generative model capable of creating artistic images from 30-second musical inputs. This contribution provides a detailed description of the generative pipeline and the dataset created for the experiment. The results obtained from this highly experimental contribution, including qualitative and quantitative outcomes, are promising and provide a solid foundation for further exploration of multimodal AI's creative potential, particularly from an emotional semantic perspective."

The dataset adopted is Museart (https://kaggle.com/datasets/64bae3c3c75ff45297b6326a1fa94114b7a1a98858cc61c09f3a36e4a8485d6b).



IMPORTANT: to reproduce the generation, download BEATs pre-trained model:

`mkdir -p models/BEATs/ && wget -O models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"`


