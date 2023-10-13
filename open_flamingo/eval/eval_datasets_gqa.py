import json
import os
from PIL import Image
from torch.utils.data import Dataset


class GQADataset(Dataset):
    def __init__(
            self,
            question_path,  # _all_questions.json
            image_folder_path,
            dataset_name,
            # dataset_range # "dev_all" or "all" or "dev_bal" or "bal"
            # ,is_train
    ):
        self.image_folder_path = image_folder_path
        # self.is_train = is_train
        self.dataset_name = dataset_name
        # self.dataset_range = dataset_range#["all", "balance", "dev_all", "dev_balance"]
        self.questions = self.load_questions(question_path)

    def load_questions(self, question_path):
        # if self.is_train:
        #     all_questions = {}
        #     for filename in os.listdir(question_path):
        #         if filename.endswith(".json"):
        #             json_file_path = os.path.join(question_path, filename)
        #             with open(json_file_path, "r") as f:
        #                 data = json.load(f)
        #                 all_questions.update(data)
        #     return all_questions
        # else:
        with open(question_path, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question_id):
        imageid = self.questions[f"{question_id}"]["imageId"]
        return os.path.join(self.image_folder_path, f"{imageid}.jpg")

    def __getitem__(self, idx):
        question_pair = self.questions
        question_id = list(question_pair.keys())[idx]

        img_path = self.get_img_path(question_id)
        image = Image.open(img_path)
        image.load()

        # new 1k test set
        question = question_pair[f"{question_id}"]["question"]
        return {
            "image": image,
            "image_id": question_pair[f"{question_id}"]["imageId"],
            "question": question,
            "question_id": question_id,
            "answer": question_pair[f"{question_id}"]["answer"],
        }


"""
        question_pair = self.questions
        question_id = list(question_pair.keys())[idx]

        img_path = self.get_img_path(question_id)
        image = Image.open(img_path)
        image.load()
        if self.dataset_range == "all" or self.dataset_range == "bal": 
            return {
                "image": image,
                "question": question_pair[f"{question_id}"]["question"],
                "question_id": question_id,
            }

        elif self.dataset_range == "dev_all" or self.dataset_range == "dev_bal":
            operations = []
            operation = question_pair[f"{question_id}"]["semantic"]
            for i in range(len(operation)):
                operations.append(operation[i]["operation"])
            return {
                "image": image,
                "image_id": question_pair[f"{question_id}"]["imageId"],
                "question": question_pair[f"{question_id}"]["question"],
                "question_id": question_id,
                "answer": question_pair[f"{question_id}"]["answer"],
                "operation": operations, 
                #"entailed": question_pair[f"{question_id}"]["entailed"],
                "fullAnswer": question_pair[f"{question_id}"]["fullAnswer"]
            }
        else:
            return None """


class VisdialDataset(Dataset):
    def __init__(
            self, image_folder_path, image_json_path, dataset_name
    ):
        self.image_folder_path = image_folder_path
        self.image_json_path = image_json_path
        self.dataset_name = dataset_name
        self.dialogs = json.load(open(image_json_path, "r"))["data"]["dialogs"]
        self.answers = json.load(open(image_json_path, "r"))["data"]["answers"]
        self.questions = json.load(open(image_json_path, "r"))["data"]["questions"]
        self.split = json.load(open(image_json_path, "r"))["split"]

    def __len__(self):
        return len(self.dialogs)

    def get_img_path(self, dialog):
        return os.path.join(self.image_folder_path, f"VisualDialog_{self.split}_{dialog['image_id']:012}.jpg")

    def __getitem__(self, idx):
        # out_dialog: dialogs at outer which includes [{image_id, dialog},...]
        out_dialog = self.dialogs[idx]
        dialog = out_dialog['dialog']  # list

        answers_space = self.answers
        # print('out_dialog',out_dialog)

        img_path = self.get_img_path(out_dialog)
        image = Image.open(img_path)
        image.load()

        results = {
            "image_id": out_dialog["image_id"],
            "image": image,
            "caption": out_dialog["caption"],
            "answers": answers_space
        }
        if self.answers is not None:
            # answers = self.answers[idx]
            # results["answers"] = [a["answer"] for a in answers["answers"]]
            for a in dialog:
                # print("a",a)
                results["question_id"] = a["question"]
                results["question"] = self.questions[a["question"]]
                if "answer" in a and a["answer"] is not None:
                    results["answer_id"] = a["answer"]
                    results["answer"] = self.answers[a["answer"]]
                if "answer_options" in a and a["answer_options"] is not None:
                    results["answer_options"] = [self.answers[i] for i in a["answer_options"]]

        return results


class VQADataset(Dataset):
    def __init__(
            self, image_folder_path, question_path, annotations_path, dataset_name
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_folder_path = image_folder_path
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_folder_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_folder_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_folder_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_folder_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        return results


def get_unique_operations(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    unique_operations = set()

    for question_id, question_info in data.items():
        semantic = question_info.get("semantic", [])
        for operation_info in semantic:
            operation = operation_info.get("operation", "")
            if operation:
                unique_operations.add(operation)

    return unique_operations


def save_operations_as_txt(main_dir, target):
    json_file_path = os.path.join(main_dir, target)
    unique_operations = get_unique_operations(json_file_path)
    sorted_operations = sorted(unique_operations)
    # print('unique_operations:',unique_operations)
    output_file_path = "gqa_sorted_test_type"
    with open(output_file_path, "w") as f:
        for operation in sorted_operations:
            f.write(operation + "\n")
    print("DONE.")


def main():
    print("DEBUG: eval_datasets_visdial.py")
    # val dataset temporarily

    main_dir = "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/"
    dataset1 = VisdialDataset(image_folder_path=os.path.join(main_dir, "visdial/VisualDialog_test2018"),
                              image_json_path=os.path.join(main_dir, "visdial/visdial_1.0_test.json"),
                              dataset_name="visdial")
    dataset2 = VQADataset(image_folder_path=os.path.join(main_dir, "MSCOCO2014/val2014"),
                          question_path=os.path.join(main_dir, "VQA_V2/v2_OpenEnded_mscoco_val2014_questions.json"),
                          annotations_path=os.path.join(main_dir, "VQA_V2/v2_mscoco_val2014_annotations.json"),
                          dataset_name="vqav2")
    dataset3 = GQADataset(image_folder_path=os.path.join(main_dir, "GQA/images"),
                          question_path=os.path.join(main_dir, "GQA/gqa_testset_1000.json"),
                          dataset_name="gqa")
    print("------------------------")
    # print('VQA TEST: DATASET 2')
    # print(dataset2[0])
    print("------------------------")
    print('GQA TEST: DATASET 3')
    print(dataset3[99])
    print(dataset3[150])
    print(dataset3[23])
    print("------------------------")

    # path ="/nfs/data2/zhang/multiinstruct/raw_datasets/visdial/visdial_1.0_test.json"
    # # Read JSON data
    # with open(path, "r") as f:
    #     data = json.load(f)

    # for i in dataset[i]["image_id"]:
    #     if i == 568676:
    #         print
    # print('image id: 568676')
    # print('caption: a woman is standing in front of a traffic light')
    # print('Q 4754, A 12426:',data["data"]["answers"][4754],data["data"]["questions"][12426])
    # print('Q 4754,A 37155:',data["data"]["answers"][4754],data["data"]["questions"][37155])
    # print('Q 5089,A 22889:',data["data"]["questions"][5089],data["data"]["questions"][22889])
    # print('Q 18784, :',data["data"]["questions"][5089],data["data"]["questions"][18784])

    # print('dataset[0]',dataset[0]["answer_options"])
    # print('dataset2[0]', dataset2[0]["answers"])
    # print("__________________________________")
    # print('dataset[0]', dataset[0]["question_id"])
    # print('dataset[0]', dataset[0]["answer_id"])
    # print("__________________________________")
    # print(f"dataset[0]['answer_options']:{dataset[0]['answer_options']}")
    # print("__________________________________")
    # print(f"dataset[0]['question']:{dataset[0]['question']}")
    # print("__________________________________")
    # #print(f"dataset[0]['question']:{dataset[0]['question']}")
    # print(f"dataset[0]['dialog'][0]['question']:{dataset[0]['dialog'][0]['question']}")
    # print("__________________________________")
    # print(f"len(batch[image']):{len(dataset[0]['image'])}")


if __name__ == "__main__":
    main()