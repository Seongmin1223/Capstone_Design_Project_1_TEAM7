
import pandas as pd

def create_look_phone_label(input_csv="driver_imgs_list.csv", output_csv="look_phone_final.csv"):
    df = pd.read_csv(input_csv)

    look_phone = ["c1", "c3"] 
    # all others = not look phone
    def map_label(cls):
        return "look_phone" if cls in look_phone else "not_look_phone"

    df["look_phone_label"] = df["classname"].apply(map_label)
    df.to_csv(output_csv, index=False)
    print(f"완료: {output_csv} 생성됨")

if __name__ == "__main__":
    create_look_phone_label()
