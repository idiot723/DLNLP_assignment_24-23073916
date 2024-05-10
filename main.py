from task.task import install_packages, load_data, generate_prompt, generate_test_prompt,load_model,predict,model_tuning
from task.task import calcu_scores, new_prompt_train_score,new_prompt_test_score, plot_test, plot_train
import pandas as pd
from datasets import Dataset

def main():

    install_packages()

    #load data
    train_df, test_df = load_data()
    train_sdf = train_df.head(2000)#train_df.head(800)
    valid_sdf = train_df.tail(2000)#train_df.head(800)
    test_sdf = test_df.head(500)
    train_test = train_df.head(500)

    #add prompt
    train = pd.DataFrame(train_sdf.apply(generate_prompt, axis=1), columns=["translations"])
    valid = pd.DataFrame(valid_sdf.apply(generate_prompt, axis=1), columns=["translations"])
    test = pd.DataFrame(test_sdf.apply(generate_test_prompt, axis=1), columns=["translations"])
    test_train = pd.DataFrame(train_test.apply(generate_test_prompt, axis=1), columns=["translations"])
    #wrapped using Hugging Face class
    train_data = Dataset.from_pandas(train)
    valid_data = Dataset.from_pandas(valid)

    #load model and text generation
    model, tokenizer = load_model("microsoft/Phi-3-mini-4k-instruct")
    y_pred1 = predict(test, model, tokenizer) #y_pred1 = predict(test_train, model, tokenizer)
    test["modern_translation"] = y_pred1 #test_train["modern_translation"] = y_pred1
    test.to_csv("./task/origin_test.csv", index=False) #test_train.to_csv("./task/origin_train.csv", index=False)

    #model tuning and text generation
    model_tuning(model,train_data,valid_data,tokenizer)
    fine_model, tokenizer = load_model("trained-model3")#"trained-model2"/"trained-model"
    y_pred2 = predict(test, fine_model, tokenizer) #y_pred2 = predict(test_train, fine_model, tokenizer)
    test["modern_translation"] = y_pred2 #test_train["modern_translation"] = y_pred2
    test.to_csv("./task/2000_testmodel.csv", index=False) #test_train.to_csv("./task/2000_model.csv", index=False)

    #evaluation
    calcu_scores(train_df,test_sdf)
    # new_prompt_train_score(train_df)
    # new_prompt_test_score(test_df)

    #plot
    plot_test("bleurt_score","./task/bleurt_score_test.png")
    plot_test("bert_score","./task/bert_score_test.png")
    plot_train("bleurt_score","./task/bleurt_score_train.png")
    plot_train("bert_score","./task/bert_score_train.png")


main()
