# Change Detection - Pix2Pix
 
Implementação do modelo de GAN **Pix2Pix** com o intuito de ser utilizada na tarefa de detecção de mudança em imagens de satélites. Implementação utilizada no projeto de trabalho de conclusão de curso na graduação de Engenharia de Computação 2021 <br/>

# Resumo
A detecção de mudanças em imagens é uma tarefa de significativo interesse para inúmeras aplicações como sensoriamento remoto, diagnóstico médico, desenvolvimento de análises urbanas e até mesmo monitoramento de processos agrícolas. Por ser uma tarefa que para ser realizada manualmente demanda muito tempo e esforço para a geração de respostas satisfatórias, a sua automatização se torna um grande campo de pesquisa. 
    
Nesse sentido, este trabalho visa a reprodução de um modelo de aprendizagem profunda desenvolvido para a detecção de mudança entre imagens de satélites. Desenvolvido no artigo "Change detection in remote sensing images using conditional adversarial networks" de autoria de M. A. Lebedev \textit{et al.}, o modelo é composto de uma rede adversária generativa capaz de considerar somente o aparecimento e desaparecimento de objetos relevantes entre duas imagens, sendo robusto o suficiente para não ser sensível à influência de fatores relacionados à variação de luminosidade e mudanças climáticas.
    
Com o intuito de analisar os resultados apresentados no trabalho original, é proposto a reprodução do modelo, teste de novos parâmetros e a discussão das contribuições do modelo para a área de detecção de mudança entre imagens.


## Referências
1. LEBEDEV, M.; VIZILTER, Y.; VYGOLOV, O.; KNYAZ, V.; RUBIS, A. Change detection inremote sensing images using conditional adversarial networks.ISPRS - International Archivesof the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLII-2, p.565–571, 05 2018.

2. ISOLA, P.; ZHU, J.-Y.; ZHOU, T.; EFROS, A. A. Image-to-image translation with conditionaladversarial networks.Computer Vision and Pattern Recognition, 2017. 
