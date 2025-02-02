# 📜 Bangladesh Premier League Analysis 

<div align="center">
    <img src="https://static.toiimg.com/thumb/msid-88446922,width-1280,height-720,resizemode-4/88446922.jpg" />
</div>

## 🎯 AIM 

To analyze player performances in the Bangladesh Premier League by extracting insights from batsmen, bowlers, and match data—ranging from toss outcomes to overall match results.

## 📊 DATASET LINK 

[https://www.kaggle.com/abdunnoor11/bpl-data](https://www.kaggle.com/abdunnoor11/bpl-data)

## 📓 KAGGLE NOTEBOOK 

[https://www.kaggle.com/code/avdhesh15/bpl-analysis](https://www.kaggle.com/code/avdhesh15/bpl-analysis)

??? Abstract "Kaggle Notebook"

    <iframe 
        src="https://www.kaggle.com/embed/avdhesh15/bpl-analysis?kernelSessionId=220268450" 
        height="600" 
        style="margin: 0 auto; width: 100%; max-width: 950px;" 
        frameborder="0" 
        scrolling="auto" 
        title="bpl-analysis">
    </iframe>

## ⚙️ TECH STACK 

| **Category**             | **Technologies**                            |
|--------------------------|---------------------------------------------|
| **Languages**            | Python                          |
| **Libraries/Frameworks** | Matplotlib, Pandas, Seaborn, Numpy |
| **Tools**                | Git, Jupyter, VS Code               |

--- 

## 📝 DESCRIPTION 

!!! info "What is the requirement of the project?"
    - This project aims to analyze player performance data from the Bangladesh Premier League (BPL) to classify players into categories such as best, good, average, and poor based on their performance.
    - The analysis provides valuable insights for players and coaches, highlighting who needs more training and who requires less, which can aid in strategic planning for future matches.

??? info "How is it beneficial and used?"
    - **For Players:** Provides feedback on their performance, helping them to improve specific aspects of their game.
    - **For Coaches:** Helps in identifying areas where players need improvement, which can be focused on during training sessions.
    - **For Team Management:** Assists in strategic decision-making regarding player selection and match planning.
    - **For Fans and Analysts:** Offers insights into player performances and trends over the league, enhancing the understanding and enjoyment of the game.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
	- Perform initial data exploration to understand the structure and contents of the dataset.
    - To learn about the topic and searching the related content like `what is league`, `About bangladesh league`, `their players` and much more.
    - Learn about the features in details by searching on the google or quora.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
	- Articles on cricket analytics from websites such as ESPNcricinfo and Cricbuzz.
    - [https://www.linkedin.com/pulse/premier-league-202223-data-analysis-part-i-ayomide-aremu-cole-iwn4e/](https://www.linkedin.com/pulse/premier-league-202223-data-analysis-part-i-ayomide-aremu-cole-iwn4e/)
    - [https://analyisport.com/insights/how-is-data-used-in-the-premier-league/](https://analyisport.com/insights/how-is-data-used-in-the-premier-league/)

--- 

## 🔍 PROJECT EXPLANATION 

### 🧩 DATASET OVERVIEW & FEATURE DETAILS 

??? example "📂 bpl.csv"

    - There are 19 features in `BPL Dataset`

    | Feature Name | Description | Datatype |
    |--------------|-------------|:--------:|
    | id | All matches unique id | int64 |
    | season | Season of the match (20XX-XX) | object |
    | match_no | Number of matches (For eg, Round 1st, 3rd, Final) | object |
    | date | Date of the match | object |
    | team_1 | Name of the first team | object |
    | team_1_score | Scoreboard of the team 1 (runs/wickets) | object |
    | team_2 | Second Team | object |
    | team_2_score | Scoreboard of the team 2 (runs/wickets) | object |
    | player_of_match | Player of the match | object |
    | toss_winner | Which team won the toss? | object |
    | toss_decision | Toss winner team decision took either 'field first' or 'bat first' | object |
    | winner | Name of the team who won the match | object |
    | venue | Venue of the match | object |
    | city | City of the match | object |
    | win_by_wickets | Team win by how many wickets | int64 |
    | win_by_runs | Team win by how many runs | int64 |
    | result | Conclusion of `win_by_wickets` & `win_by_runs` | object |
    | umpire_1 | Name of the first umpire | object |
    | umpire_2 | Name of the second umpire | object |

??? example "🛠 Developed Features from bpl.csv"

    | Feature Name | Description | Reason   | Datatype |
    |--------------|-------------|----------|:--------:|
    | team_1_run | Run scored by the team 1 | To covert `team_1_score` categorical feature into numerical feature | int64 |
    | team_1_wicket | Wickets losed by the team 1 | To covert `team_1_score` categorical feature into numerical feature | int64 |
    | team_2_run | Run scored by the team 2 | To covert `team_2_score` categorical feature into numerical feature | int64 |
    | team_2_wicket | Wickets losed by the team 2 | To covert `team_1_score` categorical feature into numerical feature | int64 |

??? example "📂 batsman.csv"

    - There are 12 features in `Batsman Dataset`

    | Feature Name | Description| Datatype |
    |--------------|------------|:--------:|
    | id | All matches unique id | int64 |
    | season | Season of the match (20XX-XX) | object |
    | match_no | Number of matches (For eg, Round 1st, 3rd, Final) | object |
    | date | Date of the match | object |
    | player_name | Player Name | object |
    | comment | How did the batsman get out? | object |
    | R | Batsman's run | int64 |
    | B | How many balls faced the batsman? | int64 |
    | M | How long their innings was in minutes? | int64 |
    | fours | No. of fours | int64 |
    | sixs | No. of sixes | int64 |
    | SR | Strike rate `(R/B)*100` | float64 |

??? example "📂 bowler.csv"

    - There are 12 features in `Bowler Dataset`

    | Feature Name | Description| Datatype |
    |--------------|------------|:--------:|
    | id | All matches unique id | int64 |
    | season | Season of the match (20XX-XX) | object |
    | match_no | Number of matches (For eg, Round 1st, 3rd, Final) | object |
    | date | Date of the match | object |
    | player_name | Player Name | object |
    | O | No. of overs bowled | float64 |
    | M | No. of middle overs bowled | int64 |
    | R | No. of runs losed | int64 |
    | W | No. of wickets secured | int64 |
    | ECON | The average number of runs they have conceded per over bowled | float64 |
    | WD | No. of wide balls | int64 |
    | NB | No. of No balls | int64 |

--- 

### 🛤 PROJECT WORKFLOW 

!!! success "Project workflow"

    ``` mermaid
      graph TD
        A[Start] --> B[Load Dataset]
        B -->|BPL Data| C[Preprocess BPL Data]
        B -->|Batsman Data| D[Preprocess Batsman Data]
        B -->|Bowler Data| E[Preprocess Bowler Data]
        
        C --> F{Generate Queries?}
        D --> F
        E --> F

        F -->|Yes| G[Graphical Visualizations]
        G --> H[Insights & Interpretation]
        H --> I[End]
        
        F -->|No| I[End]
    ```

=== "Step 1"
    - Read and explore all datasets individually.
    - Started with `bpl.csv`, analyzing its structure and features.
    - Researched dataset attributes through Google and Bangladesh cricket series data.
    - Reviewed relevant Kaggle notebooks to gain insights from existing work.

=== "Step 2"
    - Performed basic EDA to understand data distribution.
    - Identified and handled missing values.
    - Converted categorical features into numerical representations to enrich analysis.
    - Examined dataset properties using `info()` and `describe()` functions.

=== "Step 3"
    - Developed a custom function `plotValueCounts(df, col, size=(10,5))` to visualize feature distributions.
    - This function, built using Seaborn and Matplotlib, plays a key role in extracting insights.
    - Generates count plots with labeled bar values for better interpretability.

=== "Step 4"
    - Refined `bpl.csv` by merging teams with their respective divisions to prevent duplicate counting.
    - Addressed inconsistencies:
        - Resolved two tie matches recorded in the dataset.
        - Extracted team runs and wickets from match scoreboards and transformed them into numerical features.

=== "Step 5"
    - Implemented automated querying using the `plotValueCounts()` function.
    - Performed grouped analysis of winners by seasons.

#### ❓ KEY QUERIES 

- **Key Queries on `bpl.csv` Dataset:**
    - *Won the toss, took the bat first and won the match.*
    - *Won the toss, took the bat first.*
    - *Winning rate by taking `bat first` vs `field first`.*
- **Key Queries on `batsman.csv` Dataset:**
    - *Batsman who scored more than 1 century or more in a match.*
    - *Batsman who faced 50 balls or more in a match.*
    - *Batsman hit more than 10 fours.*
- **Key Queries on `bowler.csv` Dataset:**
    - *Bowler conceded 50 runs or more*
    - *Bowler took more than 4 wickets*
    - *Bowler delivered 4 wide or more*

--- 

### 🖥 CODE EXPLANATION 

=== "plotValueCounts() function"

    ```py
    import seaborn as sns
    import matplotlib.pyplot as plt

    def plotValueCounts(df, col, size=(10, 5)):
        val_counts = df[col].value_counts()
        plt.figure(figsize=size)
        ax = sns.countplot(y=col, data=df, order=val_counts.index, palette="pastel")
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', padding=3, fontsize=10, color='black')
        plt.title(f"Value Counts of {col}")
        plt.show()
    ```

    - It displays the visualization graph of value counts of any feature of the dataset.

--- 

### ⚖️ PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"
    - **Trade-off:** The dataset includes multiple sources, but some features required external validation (e.g., team names, score formats).
    - **Solution:** Standardized team names and resolved inconsistencies by merging duplicate team entries based on divisions.

=== "Trade Off 2"
    - **Trade-off:** Performing detailed feature engineering (e.g., extracting runs, wickets, and match insights) can increase computational overhead.
    - **Solution:** Optimized preprocessing by transforming scoreboard data into structured numerical features, reducing redundant calculations.

=== "Trade Off 3"
    - **Trade-off:** Count plots with high-cardinality features could lead to cluttered or unreadable visualizations.
    - **Solution:** Implemented `plotValueCounts()` with bar labels and sorting to enhance clarity while retaining key insights.

=== "Trade Off 4"
    - **Trade-off:** Writing fixed queries could limit adaptability when new data is introduced.
    - **Solution:** Developed a function-based querying approach (e.g., `plotValueCounts(df, col)`) to enable flexible, real-time insights.

--- 

## 🖼 SCREENSHOTS 

!!! tip "Visualizations and EDA of different features"

    === "Toss Winner"
        ![toss_winner](https://github.com/user-attachments/assets/bf2eb957-1a6c-4c00-8189-5eb26f6c0af6)

    === "Winner Teams"
        ![winner](https://github.com/user-attachments/assets/f7eef68c-b2a8-40e1-bec5-b7cdff040b3e)

    === "Win Match By Wickets"
        ![win_match_by_wickets](https://github.com/user-attachments/assets/5f1363ce-4812-45f2-b218-1dece5c8a227)

    === "Win Match By Runs"
        ![win_match_by_runs](https://github.com/user-attachments/assets/6da68090-3e33-4a9a-9200-d07d7dd6e067)

    === "winners By Season"
        ![winners_by_season](https://github.com/user-attachments/assets/c21bc48b-69e9-4c67-aa82-571644202ef6)

    === "Venue of Match"
        ![venue_of_match](https://github.com/user-attachments/assets/76dce847-3347-48b5-956a-041b3b166b4e)

    === "Top 10 Player of Match"
        ![top_10_player_of_match](https://github.com/user-attachments/assets/feba3f7e-3fd3-47a5-8ec3-6c676b4119f3)

    === "Winning Rate By Toss Decision"
        ![winning_rate_by_toss_decision](https://github.com/user-attachments/assets/2868decc-62a3-4d79-84d8-4bdb9aad27c9)
    
    === "Batsman scored more than 1 century"
        ![batsman_more_than_1_century](https://github.com/user-attachments/assets/fe5c81ce-909d-4d9f-a796-5f6a86e0df81)
    
    === "Bowler took more than 4 wickets"
        ![bowler_took_more_4_wickets](https://github.com/user-attachments/assets/6cf04e8a-76e1-401a-b0b0-0a8fc4a25a65)

--- 

## ✅ CONCLUSION 

### 🔑 KEY LEARNINGS 

!!! tip "Insights gained from the data"
    - Extracted meaningful player statistics by analyzing batting and bowling performances across multiple seasons.
    - Identified trends in match outcomes based on factors like toss decisions, innings strategies, and scoring patterns.
    - Enhanced data visualization techniques to present key insights effectively for better decision-making.

--- 

### 🌍 USE CASES 

=== "Application 1"
	**Team Strategy & Selection**
     - Helps coaches analyze player performance for better team selection and match strategies.

=== "Application 2"
	**Player Performance Tracking**
     - Assists in monitoring player trends to improve training and development.
