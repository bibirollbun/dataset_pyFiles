# imports 
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
from tqdm.notebook import tqdm, trange

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

tqdm.pandas()


# Util Functions

def create_nfl_field() -> np.ndarray:
    """creates a simplified image of a football field as a background for rendering

    Returns:
        np.ndarray: the image in form of a np.array of the shape (x, y, 3)
    """
    empty_field = np.zeros((533, 1200, 3), dtype=np.int16) + np.array([50, 130, 20])
    empty_field[:, 0:100] = 127
    empty_field[:, 1100:] = 127
    empty_field[:, 90:101] = 255
    empty_field[:, 1100:1111] = 255
    for i in range(150, 1051, 50):
        empty_field[:, i - 2 : i + 3] = 255
    for i in range(110, 1091, 10):
        empty_field[:10, i - 1 : i + 2] = 255
        empty_field[-10:, i - 1 : i + 2] = 255
        empty_field[197:208, i - 1 : i + 2] = 255
        empty_field[327:338, i - 1 : i + 2] = 255
    return empty_field

def fuzzy_intersection(*args, acceptance_threshold=2):
    ### assume args are sorted lists
    result = []

    for x in args[0]:
        found_in_all = True
        for lst in args[1:]:
            i = 0
            while i < len(lst) and lst[i] < x - acceptance_threshold:
                i += 1
            if i == len(lst) or abs(lst[i] - x) > acceptance_threshold:
                found_in_all = False
                break
        if found_in_all:
            result.append(x)

    return result

def get_cut_timing(df, n=1, omega_thres=.05, min_time=5): 
    min_speed = argrelextrema(df["v"].rolling(3).mean().values, np.less, order=n)[0]
    max_accel = argrelextrema(df["a"].rolling(3).mean().values, np.greater, order=n)[0]
    
    ts = df["angle_speed"].rolling(3).mean().values
    ts[np.fabs(ts)< omega_thres] = 0
    min_omega = argrelextrema(ts, np.less, order=n)[0]
    max_omega = argrelextrema(ts, np.greater, order=n)[0]
    
    min_speed = min_speed[min_speed > min_time]
    
    cut_t = fuzzy_intersection(min_speed, max_accel, np.hstack([min_omega, max_omega]))
    if len(cut_t) == 1: 
        return cut_t[0]
    return None


BASE_DIR = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"

supp = pd.read_csv(f"{BASE_DIR}/supplementary_data.csv")
weeks_in = [pd.read_csv(f"{BASE_DIR}/train/input_2023_w{i:02d}.csv") for i in trange(1,19)]
tracking_in = pd.concat([wk for wk in weeks_in], ignore_index=True)
weeks_out = [pd.read_csv(f"{BASE_DIR}/train/output_2023_w{i:02d}.csv") for i in trange(1,19)]
tracking_out = pd.concat([wk for wk in weeks_out], ignore_index=True)


colors = {"Defense": "lime", "Offense":"navy"}

tracking_in["dist_to_land"] = np.sqrt((tracking_in["x"]-tracking_in["ball_land_x"])**2 + (tracking_in["y"]-tracking_in["ball_land_y"])**2)

tracking_in = pd.merge(
    tracking_in, 
    tracking_in.loc[(tracking_in["player_role"]=="Targeted Receiver") & ~(tracking_in.duplicated(["game_id", "play_id", "nfl_id"], keep="last")), ["game_id", "play_id", "frame_id", "dist_to_land"]].rename(columns={"frame_id":"throw_frame", "dist_to_land":"rec_dist_at_throw"}),
    on=["game_id", "play_id"], 
    how="left")

tracked_players = tracking_out.groupby(["game_id", "play_id"]).nfl_id.nunique().reset_index()
throw_frames = tracking_in[["game_id", "play_id", "frame_id"]].sort_values(["game_id", "play_id", "frame_id"], ascending=True).drop_duplicates(["game_id", "play_id"],keep="last")
tracking_out["air_frame"] = tracking_out.frame_id.values
tracking_out["frame_id"] = tracking_out["frame_id"] + tracking_out[["game_id", "play_id"]].merge(throw_frames, on=["game_id", "play_id"], how="left")["frame_id"]
tracking_out = tracking_out.merge(tracking_in[["game_id", "play_id", "nfl_id", "player_name", "player_role", "player_position", "player_side", "ball_land_x", "ball_land_y", "throw_frame", "play_direction", "absolute_yardline_number"]].drop_duplicates(["game_id", "play_id", "nfl_id"], keep="last"), how="left", on=["game_id", "play_id", "nfl_id"])
tracking_in["air_frame"] = -1 
tracking_out["player_to_predict"] = 1
tracking_full = pd.concat([tracking_in[tracking_out.columns], tracking_out], axis=0, ignore_index=True).sort_values(["game_id", "play_id", "nfl_id", "frame_id"], ascending=True)



tracking_full = tracking_full.merge(tracked_players[tracked_players.nfl_id>1][['game_id','play_id']], on=["game_id", "play_id"], how="inner")


tracking_full[["xs", "ys"]] = tracking_full.groupby(["game_id", "play_id", "nfl_id"])[["x", "y"]].rolling(3, min_periods=1, center=True).mean().reset_index().set_index("level_3")[["x", "y"]]


# change in position / timestep ~ velocity
tracking_full["vx"] = tracking_full["xs"].diff()
tracking_full.loc[(tracking_full.nfl_id.diff() != 0) | (tracking_full.play_id.diff() != 0), "vx"] = 0.0
tracking_full["vy"] = tracking_full["ys"].diff()
tracking_full.loc[(tracking_full.nfl_id.diff() != 0) | (tracking_full.play_id.diff() != 0), "vy"] = 0.0

# change in velocity / timestep ~ acceleration
tracking_full["ax"] = tracking_full["vx"].diff()
tracking_full.loc[(tracking_full.nfl_id.diff() != 0) | (tracking_full.play_id.diff() != 0), "ax"] = 0.0
tracking_full["ay"] = tracking_full["vy"].diff()
tracking_full.loc[(tracking_full.nfl_id.diff() != 0) | (tracking_full.play_id.diff() != 0), "ay"] = 0.0

# scalar value of movement velocity and acceleration
tracking_full["v"] = np.sqrt(tracking_full["vx"]**2 + tracking_full["vy"]**2)
tracking_full["a"] = np.sqrt(tracking_full["ax"]**2 + tracking_full["ay"]**2)


tracking_full["dist_to_land"] = np.sqrt((tracking_full["xs"]-tracking_full["ball_land_x"])**2 + (tracking_full["ys"]-tracking_full["ball_land_y"])**2)


# merge targeted receiver details as new columns
tracking_full = pd.merge(
    tracking_full, 
    tracking_full.loc[(tracking_full["player_role"]=="Targeted Receiver"), ["game_id", "play_id", "frame_id", "xs","ys", "v", "a", "dist_to_land"]].rename(columns={"xs":"rec_x", "ys":"rec_y", "a":"rec_a", "v":"rec_v", "dist_to_land":"rec_dist_to_land"}),
    on=["game_id", "play_id", "frame_id"], 
    how="left")


tracking_full["rec_separation"] = np.sqrt((tracking_full["xs"] - tracking_full["rec_x"])**2 + (tracking_full["ys"] - tracking_full["rec_y"])**2)



tracking_full = tracking_full.merge(supp[["game_id", "play_id", "pass_result"]], on=["game_id", "play_id"])
tracking_full[["C", "I", "IN"]] = pd.get_dummies(tracking_full.pass_result)


tracking_full["movement_dir"] = np.arctan2(tracking_full.vy, tracking_full.vx )
tracking_full.loc[(tracking_full.nfl_id.diff() != 0) | (tracking_full.play_id.diff() != 0), "movement_dir"] = pd.NA
tracking_full.movement_dir.bfill()
tracking_full["angle_speed"] = tracking_full["movement_dir"].diff()
tracking_full["angle_speed"] = (tracking_full["angle_speed"] + np.pi) % (2*np.pi) - np.pi
tracking_full.loc[(tracking_full.nfl_id.diff() != 0) | (tracking_full.play_id.diff() != 0), "angle_speed"] = 0.0


cut_times = tracking_full[tracking_full.player_role=="Targeted Receiver"].groupby(['game_id', 'play_id', 'nfl_id']).progress_apply(get_cut_timing)


supp.merge(cut_times[cut_times.notna()].reset_index(), on=["game_id", "play_id"], how="inner").route_of_targeted_receiver.value_counts()



smpl = supp[(supp.game_id==2023091800)& (supp.play_id==1027)].merge(
    cut_times[cut_times.notna()].reset_index().rename(columns={0:"cut_frame"}),
    on=["game_id", "play_id"], how="inner").sample()[
        ["game_id", "play_id", "yardline_number", "yards_to_go", "yardline_side", "defensive_team", "play_description", "quarter", "week", "home_team_abbr", "visitor_team_abbr", "cut_frame", "route_of_targeted_receiver"]
        ]
smpldf = tracking_full.merge(smpl, on=["game_id", "play_id"], how="inner")



rec_id = smpldf[smpldf.player_role=="Targeted Receiver"].drop_duplicates("nfl_id", keep="last").nfl_id.iat[0]
db_id = smpldf[smpldf.player_role=="Defensive Coverage"].drop_duplicates("nfl_id", keep="last").sort_values("rec_separation", ascending=True).nfl_id.iat[0]

ydn, ytg, ydside, dteam,cut_frame = smpl[["yardline_number", "yards_to_go", "yardline_side", "defensive_team", "cut_frame"]].values[0]


ttdf_in = smpldf[(smpldf.air_frame.lt(0))]
ttdf_out = smpldf[(smpldf.air_frame.ge(0))]


throw_frame = smpldf.throw_frame.iat[0]
max_frame = smpldf.frame_id.max()
 


if ydside == dteam:
    ydn = 100-ydn

if ttdf_in.play_direction.iat[0]=='right': 
    los = ydn*10 + 100
    newdowns = los + ytg*10 
else:
    los = 1200-100-ydn*10
    newdowns = los-ytg*10

with plt.style.context({"font.size":18}):
    fig, ax = plt.subplots(2,2,figsize=(12,9))

    ax[0,0].plot(smpldf[smpldf.nfl_id==rec_id].frame_id,smpldf[smpldf.nfl_id==db_id].rec_separation)
    ax[0,0].set_ylim(0, smpldf[smpldf.nfl_id==db_id].rec_separation.max()*1.2)
    ax[0,1].plot(smpldf[smpldf.nfl_id==rec_id].frame_id, smpldf[smpldf.nfl_id==rec_id].v, label="Speed", color="navy")
    ax[0,1].plot(smpldf[smpldf.nfl_id==rec_id].frame_id, smpldf[smpldf.nfl_id==rec_id].a, label="Accel", linestyle="--", color="navy")
    ax[1,0].plot(smpldf[smpldf.nfl_id==rec_id].frame_id, smpldf[smpldf.nfl_id==rec_id].movement_dir*np.pi/180, label="WR", color="navy")
    ax[1,1].plot(smpldf[smpldf.nfl_id==rec_id].frame_id, smpldf[smpldf.nfl_id==rec_id].angle_speed, label="WR", color="navy")

    ax[0,0].set_ylabel("Receiver separation")
    ax[0,1].set_ylabel("Speed / Accel")
    ax[1,0].set_ylabel(r"$\theta$")
    ax[1,1].set_ylabel(r"$\omega$")

    for ax_ in ax:
        for axi in ax_:  
            axi.axvline(throw_frame, alpha=.3, label=r"$t_t$")
            axi.axvline(cut_frame, alpha=.3, linestyle=":", label=f"$t_c$")
            axi.set_xlabel("t")
            axi.legend(loc='upper left')


    fig.tight_layout()
    plt.show()

    ####################################

    fig, ax = plt.subplots(1,1,figsize=(20,12))
    ax.imshow(create_nfl_field(), origin="lower", alpha=.4)

    plt.vlines([los, newdowns], color=["blue", "orange"], ymin=0, ymax=533.3, linewidth=5, alpha=.8)


    bx, by = smpldf[["ball_land_x", "ball_land_y"]].values[0]*10
    targeted_rec_x, targeted_rec_y = smpldf.loc[smpldf.player_role == "Targeted Receiver", ("xs", "ys")].values[0]*10


    plt.scatter([bx], [by], color="brown", s=100, marker='D', label='Ball landing')

    plt.scatter(ttdf_in[ttdf_in.player_role!="Defensive Coverage"]["xs"].values*10, ttdf_in[ttdf_in.player_role!="Defensive Coverage"]["ys"].values*10, 
                marker="x", 
                c=ttdf_in[ttdf_in.player_role!="Defensive Coverage"]['player_side'].map(colors), 
                label='Before throw, WR'
            )
    plt.scatter(ttdf_in[ttdf_in.player_role=="Defensive Coverage"]["xs"].values*10, ttdf_in[ttdf_in.player_role=="Defensive Coverage"]["ys"].values*10, 
                marker="x", 
                c=ttdf_in[ttdf_in.player_role=="Defensive Coverage"]['player_side'].map(colors), 
                label='Before throw, DB'
            )

    plt.scatter(ttdf_out[ttdf_out.player_role=="Targeted Receiver"]["xs"].values*10, ttdf_out[ttdf_out.player_role=="Targeted Receiver"]["ys"].values*10, marker="o", color="teal", label="Ball airborne, WR")
    plt.scatter(ttdf_out[ttdf_out.player_role=="Defensive Coverage"]["xs"].values*10, ttdf_out[ttdf_out.player_role=="Defensive Coverage"]["ys"].values*10, marker="o", color="gold", label="Ball airborne, DB")

    plt.scatter(smpldf[smpldf.frame_id==cut_frame]["xs"]*10, smpldf[smpldf.frame_id==cut_frame]["ys"]*10, color="brown", label="At cut")
    
    plt.title('')
    plt.legend()
    plt.ylim(0,533)
    plt.xlim(0,1200)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print(f"{smpl.visitor_team_abbr.iat[0]} @ {smpl.home_team_abbr.iat[0]}, week {smpl.week.iat[0]}, quarter {smpl.quarter.iat[0]}, Route:{smpl.route_of_targeted_receiver.iat[0]}")
    print("Result:", smpldf.pass_result.iat[0])
    print(smpl.play_description.iat[0])


cut_df = pd.merge(cut_times.reset_index().rename(columns={0:"cut_time"}), throw_frames.reset_index().rename(columns={"frame_id":"throw_frame"}), on=["game_id", "play_id"], how="inner")
cut_df["delta_t"] = cut_df["cut_time"] - cut_df["throw_frame"]
cut_df = pd.merge(cut_df[cut_df.cut_time.notna()], tracking_full[tracking_full.player_role=="Targeted Receiver"].drop_duplicates(["game_id", "play_id"], keep="last")[["game_id", "play_id", "frame_id", "air_frame", "C", "I", "IN"]], on=["game_id", "play_id"], how="inner")
cut_df = cut_df.merge(tracking_full.loc[tracking_full.player_role=="Defensive Coverage",["game_id", "play_id", "frame_id", "rec_separation"]].sort_values("rec_separation").drop_duplicates(["game_id", "play_id", "frame_id"], keep="first").rename(columns={"rec_separation":"final_sep"}), left_on=["game_id", "play_id", "frame_id"], right_on=["game_id", "play_id", "frame_id"])
cut_df = cut_df.merge(tracking_full.loc[tracking_full.player_role=="Defensive Coverage",["game_id", "play_id", "frame_id", "rec_separation"]].sort_values("rec_separation").drop_duplicates(["game_id", "play_id", "frame_id"], keep="first").rename(columns={"rec_separation":"sep_at_cut"}), left_on=["game_id", "play_id", "cut_time"], right_on=["game_id", "play_id", "frame_id"])
cut_df = cut_df.merge(tracking_full.loc[tracking_full.player_role=="Defensive Coverage",["game_id", "play_id", "frame_id", "rec_separation"]].sort_values("rec_separation").drop_duplicates(["game_id", "play_id", "frame_id"], keep="first").rename(columns={"rec_separation":"sep_at_throw"}), left_on=["game_id", "play_id", "throw_frame"], right_on=["game_id", "play_id", "frame_id"])
cut_df = cut_df.merge(tracking_full.loc[(tracking_full.player_role=="Defensive Coverage") & (tracking_full["frame_id"]==1),["game_id", "play_id", "rec_separation"]].sort_values("rec_separation").drop_duplicates(["game_id", "play_id"], keep="first").rename(columns={"rec_separation":"sep_at_snap"}), left_on=["game_id", "play_id"], right_on=["game_id", "play_id"])


def avg_rec_sep(df):
    db_id = df[(df.player_role=="Defensive Coverage") & (df.frame_id>df.throw_frame)].sort_values("rec_separation", ascending=True).nfl_id.iat[0]
    return df[(df.nfl_id == db_id) & (df.frame_id>df.throw_frame)].rec_separation.mean()
    
def avg_d2l(df):
    db_id = df[(df.player_role=="Defensive Coverage") & (df.frame_id>df.throw_frame)].drop_duplicates("nfl_id", keep="last").sort_values("rec_separation", ascending=True).nfl_id.iat[0]
    return df[(df.nfl_id == db_id) & (df.frame_id>df.throw_frame)].dist_to_land.mean()/df[(df.nfl_id == db_id) & (df.frame_id>df.throw_frame)].rec_dist_to_land.mean()

def ex_sep_func(df):
    if "Defensive Coverage" in df.player_role.values:
        db_id = df[(df.player_role == "Defensive Coverage") & (df.frame_id == df.frame_id.max())].drop_duplicates("nfl_id", keep="last").sort_values("rec_separation", ascending=True).nfl_id.iat[0]
        return df[df.nfl_id==db_id].sort_values("frame_id", ascending=True).rec_separation.values
    return None


avg_sep = tracking_full[tracking_full["player_role"]=="Defensive Coverage"].groupby(["game_id", "play_id"]).progress_apply(avg_rec_sep).reset_index().rename(columns={0:"avg_sep"})


avg_dist_to_land = tracking_full.groupby(["game_id", "play_id"]).progress_apply(avg_d2l).reset_index().rename(columns={0:"rec_rel_dist_to_land"})     



cut_df = pd.merge(cut_df, avg_sep, on=["game_id", "play_id"], how="inner")
cut_df = pd.merge(cut_df, avg_dist_to_land, on=["game_id", "play_id"], how="inner")



v_trajectories = tracking_full[tracking_full.player_role=="Targeted Receiver"].groupby(["game_id", "play_id"]).progress_apply(lambda df: df.v.values).reset_index().rename(columns={0:"v_traj"})
a_trajectories = tracking_full[tracking_full.player_role=="Targeted Receiver"].groupby(["game_id", "play_id"]).progress_apply(lambda df: df.a.values).reset_index().rename(columns={0:"a_traj"})
omega_trajectories =tracking_full[tracking_full.player_role=="Targeted Receiver"].groupby(["game_id", "play_id"]).progress_apply(lambda df: df.angle_speed.fillna(0.).values).reset_index().rename(columns={0:"omega_traj"})
abs_omega_trajectories =tracking_full[tracking_full.player_role=="Targeted Receiver"].groupby(["game_id", "play_id"]).progress_apply(lambda df: np.fabs(df.angle_speed.fillna(0.).values)).reset_index().rename(columns={0:"abs_omega_traj"})


cut_df = pd.merge(cut_df, v_trajectories, on=["game_id", "play_id"], how="inner")
cut_df = pd.merge(cut_df, a_trajectories, on=["game_id", "play_id"], how="inner")
cut_df = pd.merge(cut_df, omega_trajectories, on=["game_id", "play_id"], how="inner")
cut_df = pd.merge(cut_df, abs_omega_trajectories, on=["game_id", "play_id"], how="inner")


cut_df["agg_dir_change"] = cut_df.progress_apply(lambda df: df.abs_omega_traj[int(df.cut_time-10):int(df.cut_time+10)].sum(), axis=1)



X = pd.DataFrame(
    {
        "sep_at_snap" : (cut_df["sep_at_snap"].to_numpy()-cut_df["sep_at_snap"].mean())/cut_df["sep_at_snap"].std(),
        "sep_at_cut" : (cut_df["sep_at_cut"].to_numpy()-cut_df["sep_at_cut"].mean())/cut_df["sep_at_cut"].std(),
        "sep_at_throw" : (cut_df["sep_at_throw"].to_numpy()-cut_df["sep_at_throw"].mean())/cut_df["sep_at_throw"].std(),
        "cut_prog" : (cut_df["cut_time"]/cut_df["frame_id_x"]).to_numpy(),
        "play_duration" : (cut_df["frame_id_x"]/10).to_numpy(),
        "agg_omega" : (cut_df["agg_dir_change"].to_numpy()-cut_df["agg_dir_change"].mean())/cut_df["agg_dir_change"].std(),
        "after_throw" : (cut_df["delta_t"]>0).to_numpy(),
        "delta_t" : (cut_df["delta_t"]/10).to_numpy(),

    }
)



y_scalar = (cut_df["final_sep"].to_numpy()-cut_df["final_sep"].mean())/cut_df["final_sep"].std()

y_rel = (cut_df["final_sep"]-cut_df["sep_at_cut"]).to_numpy()
y_rel_mean = y_rel.mean()
y_rel_std = y_rel.std()
y_rel = (y_rel-y_rel.mean())/y_rel.std()

y_ratio = cut_df["rec_rel_dist_to_land"].to_numpy().clip(0,5)




X.corrwith(cut_df["final_sep"]-cut_df["sep_at_cut"])



plot_col = 'final_sep'
xmin = 0
xmax = 5
mus = cut_df.groupby(np.round(cut_df[plot_col],1))[["C", "I", "IN"]].mean()
with plt.style.context({"font.size": 18}):
    fig, (ax, ax1) = plt.subplots(2,1, height_ratios=[3,1], figsize=(12,10))
    mus.plot(ax=ax)
    ax.hlines([cut_df.C.mean()], xmin=xmin, xmax=xmax, color="b", alpha=.7, linestyle="--")
    ax.hlines([cut_df.I.mean()], xmin=xmin, xmax=xmax, color="r", alpha=.7, linestyle="--")
    ax.hlines([cut_df.IN.mean()], xmin=xmin, xmax=xmax, color="g", alpha=.7, linestyle="--")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(plot_col)
    ax.set_ylabel("%")
    ax.set_xlabel("Separation at ball arrival [yd]")
    ax.legend(loc="upper left", ncols=3)            
    cut_df[cut_df[plot_col].le(xmax)][plot_col].plot.hist(bins=51, ax=ax1)
    ax1.set_xlabel("Separation at ball arrival [yd]")
    plt.xlim(xmin, xmax)
    plt.show()


plot_col = 'avg_sep'
xmin = 0
xmax = 5
mus = cut_df.groupby(np.round(cut_df[plot_col],1))[["C", "I", "IN"]].mean()
with plt.style.context({"font.size": 18}):
    fig, (ax, ax1) = plt.subplots(2,1, height_ratios=[3,1], figsize=(12,10))
    mus.plot(ax=ax)
    ax.hlines([cut_df.C.mean()], xmin=xmin, xmax=xmax, color="b", alpha=.7, linestyle="--")
    ax.hlines([cut_df.I.mean()], xmin=xmin, xmax=xmax, color="r", alpha=.7, linestyle="--")
    ax.hlines([cut_df.IN.mean()], xmin=xmin, xmax=xmax, color="g", alpha=.7, linestyle="--")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(plot_col)
    ax.set_xlabel("Average separation while ball is airborne [yd]")
    ax.legend(loc="upper left", ncols=3)            
    cut_df[cut_df[plot_col].le(xmax)][plot_col].plot.hist(bins=51, ax=ax1)
    ax1.set_xlabel("Average separation while ball is airborne [yd]")
    plt.xlim(xmin, xmax)
    plt.show()



plot_col = 'rec_rel_dist_to_land'
xmin = 0
xmax = 5
mus = cut_df.groupby(np.round(cut_df[plot_col],1))[["C", "I", "IN"]].mean()
with plt.style.context({"font.size": 18}):
    fig, (ax, ax1) = plt.subplots(2,1, height_ratios=[3,1], figsize=(12,10))
    mus.plot(ax=ax)
    ax.hlines([cut_df.C.mean()], xmin=xmin, xmax=xmax, color="b", alpha=.7, linestyle="--")
    ax.hlines([cut_df.I.mean()], xmin=xmin, xmax=xmax, color="r", alpha=.7, linestyle="--")
    ax.hlines([cut_df.IN.mean()], xmin=xmin, xmax=xmax, color="g", alpha=.7, linestyle="--")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Air Leverage")
    ax.set_ylabel("%")
                
    
    cut_df[cut_df[plot_col].le(xmax)][plot_col].plot.hist(bins=51, ax=ax1)
    plt.xlim(xmin, xmax)
    ax.legend(loc="upper left", ncols=3)
    ax1.set_xlabel("Air Leverage")
    plt.show()


X_train, X_test, y_train, y_test, y_train2, y_test2 = train_test_split(
    X, y_rel, y_ratio,  test_size=0.2
)


rf_reg = RandomForestRegressor(n_estimators=1500)


rf_reg.fit(X_train, y_train)



rf_pred = rf_reg.predict(X_test)

print("Air Leverage R^2:", rf_reg.score(X_test, y_test))
print(f"Air Leverage MSE: {mean_squared_error(y_test, rf_pred)}")
print(f"Air Leverage MAE: {mean_absolute_error(y_test, rf_pred)}")


pd.DataFrame({"fname":rf_reg.feature_names_in_, "importance":rf_reg.feature_importances_}).sort_values("importance", ascending=False)



tix = 44
print(f"predicted: {rf_pred[tix]:.2f}, actual: {y_test[tix]:.2f}")
print("Data:", X_test.iloc[tix])
explainer = shap.TreeExplainer(rf_reg)
choosen_instance = X_test.iloc[tix]
shap_values = explainer.shap_values(X_test.iloc[:50])
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[tix], choosen_instance)


res = permutation_importance(rf_reg, X_test, y_test, n_repeats=11)
fig, ax = plt.subplots(figsize=(10,8))
forest_importances = pd.Series(res.importances_mean, index=X_test.columns.to_list())
forest_importances.plot.bar(yerr=res.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()


with plt.style.context({"font.size":18}):
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.hist2d(rf_pred,y_test, bins=50, alpha=.93)
    ax.set_xlabel("Predicted change in separation")
    ax.set_ylabel("Actual change in separation")
    ax.plot([-5,5], [-5,5], color="lime", linestyle="--", alpha=.8, label="Predicted=Actual")
    plt.legend(loc="upper left")
    plt.show()


for col, clabel in zip(["sep_at_cut", "sep_at_throw", "delta_t"], [r"$Separation\ at\ t_c$  [yd]", r"$Separation\ at\ t_t$ [yd]", r"$t_c - t_t$ [s]"]):
    with plt.style.context({"font.size":18}):
        plt.figure(figsize=(12,10))
        all_vals = []
        for _ in trange(200): 
            smpl_dict = X_test.sample().iloc[0].to_dict()
            x_vals = np.linspace(X[col].quantile(.05), X[col].quantile(0.95),100)
            inpts=[]
            for sac in x_vals:
                smpl_dict[col] = sac
                inpts.append(smpl_dict.copy())

            vals = rf_reg.predict(pd.DataFrame(inpts))
            all_vals.append(vals)
        mu_vals = np.mean(all_vals, axis=0)
        mu_p05 = np.quantile(all_vals, [0.16, 0.84], axis=0)

        y_ratio_std = y_rel_std
        y_ratio_mean = y_rel_mean

        plt.plot((x_vals*cut_df[col].std())+cut_df[col].mean(),  (mu_vals*y_ratio_std)+y_ratio_mean, label="Mean")
        plt.fill_between((x_vals*cut_df[col].std())+cut_df[col].mean(), (mu_p05[0]*y_ratio_std)+y_ratio_mean, (mu_p05[1]*y_ratio_std)+y_ratio_mean, alpha=.2, label=r"$\mu \pm \sigma$ CI")
        plt.xlabel(clabel)
        plt.ylabel("Change in Separation [yd]")
        plt.axhline(0., alpha=.3, linestyle="--")
        plt.legend(loc = "upper left")
        plt.show()


rf_reg2 = RandomForestRegressor(n_estimators=1500)


rf_reg2.fit(X_train, y_train2)



rf_pred2 = rf_reg2.predict(X_test)

print("R^2:", rf_reg2.score(X_test, y_test2))
print(f"MSE: {mean_squared_error(y_test2, rf_pred2)}")
print(f"MAE: {mean_absolute_error(y_test2, rf_pred2)}")


pd.DataFrame({"fname":rf_reg2.feature_names_in_, "importance":rf_reg2.feature_importances_}).sort_values("importance", ascending=False)



tix = 44
print(f"predicted: {rf_pred[tix]:.2f}, actual: {y_test[tix]:.2f}")
print("Data:", X_test.iloc[tix])
explainer = shap.TreeExplainer(rf_reg2)
choosen_instance = X_test.iloc[tix]
shap_values = explainer.shap_values(X_test.iloc[:50])
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[tix], choosen_instance)


res = permutation_importance(rf_reg2, X_test, y_test2, n_repeats=11)
with plt.style.context({"font.size":18}):
    fig, ax = plt.subplots(figsize=(10,8))
    forest_importances = pd.Series(res.importances_mean, index=X_test.columns.to_list())
    forest_importances.plot.bar(yerr=res.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


with plt.style.context({"font.size":18}):
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.hist2d(rf_pred2,y_test2, bins=50, alpha=.93)
    ax.set_xlabel("Predicted Air Leverage")
    ax.set_ylabel("Actual Air Leverage")
    ax.plot([1,5], [1,5], color="lime", linestyle="--", alpha=.8, label="Predicted=Actual")
    plt.legend(loc="upper left")
    plt.title("Air Leverage Model Performance")
    plt.show()


for col, clabel in zip(["sep_at_throw", "sep_at_cut", "delta_t", "agg_omega", "cut_prog"], [r"$Separation\ at\ t_t$ [yd]", r"$Separation\ at\ t_c$ [yd]", r"$t_c - t_t$ [s]", "aggregated change in radians", r"$t_c / t_a$"]):
    with plt.style.context({"font.size":18}):
        plt.figure(figsize=(12,10))
        all_vals = []
        for _ in trange(200): 
            smpl_dict = X_test.sample().iloc[0].to_dict()
            x_vals = np.linspace(X[col].quantile(.005), X[col].quantile(0.995),100)
            inpts=[]
            for sac in x_vals:
                smpl_dict[col] = sac
                inpts.append(smpl_dict.copy())

            vals = rf_reg2.predict(pd.DataFrame(inpts))
            all_vals.append(vals)
        mu_vals = np.mean(all_vals, axis=0)
        mu_p05 = np.quantile(all_vals, [0.16, 0.84], axis=0)

        y_ratio_std = 1
        y_ratio_mean = 0

        if col == "agg_omega": 
            plt.plot((x_vals*cut_df["agg_dir_change"].std())+cut_df["agg_dir_change"].mean(),  (mu_vals*y_ratio_std)+y_ratio_mean, label="Mean")
            plt.fill_between((x_vals*cut_df["agg_dir_change"].std())+cut_df["agg_dir_change"].mean(), (mu_p05[0]*y_ratio_std)+y_ratio_mean, (mu_p05[1]*y_ratio_std)+y_ratio_mean, alpha=.2, label=r"$\mu \pm \sigma \ CI$")
        elif col == "cut_prog": 
            plt.plot(x_vals,  (mu_vals*y_ratio_std)+y_ratio_mean, label="Mean")
            plt.fill_between(x_vals, (mu_p05[0]*y_ratio_std)+y_ratio_mean, (mu_p05[1]*y_ratio_std)+y_ratio_mean, alpha=.2, label=r"$\mu \pm \sigma \ CI$")
        else:
            plt.plot((x_vals*cut_df[col].std())+cut_df[col].mean(),  (mu_vals*y_ratio_std)+y_ratio_mean, label="Mean")
            plt.fill_between((x_vals*cut_df[col].std())+cut_df[col].mean(), (mu_p05[0]*y_ratio_std)+y_ratio_mean, (mu_p05[1]*y_ratio_std)+y_ratio_mean, alpha=.2, label=r"$\mu \pm \sigma \ CI$")
        plt.xlabel(clabel)
        plt.ylabel("Air Leverage")
        plt.axhline(np.median(y_ratio), alpha=.3, linestyle="--")
        plt.legend(loc = "upper left")
        plt.show()




