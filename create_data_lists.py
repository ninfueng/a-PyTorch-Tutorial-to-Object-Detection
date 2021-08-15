from utils import create_data_lists

if __name__ == "__main__":
    #create_data_lists(
    #    voc07_path="/home/tm_ninnart16/datasets/voc/VOCdevkit/VOC2007",
    #    voc12_path="/home/tm_ninnart16/datasets/voc/VOCdevkit/VOC2012",
    #    output_folder="./",
    #)
    create_data_lists(
        voc07_path="/groups/gcc50527/ninnart/datasets/VOCdevkit/VOC2007",
        voc12_path="/groups/gcc50527/ninnart/datasets/VOCdevkit/VOC2012",
        output_folder="./",
    )
