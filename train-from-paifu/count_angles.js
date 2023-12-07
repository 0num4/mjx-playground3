function countAngles(angles) {
    let count90 = 0, count45 = 0, count0 = 0, count135 = 0;

    angles.forEach(angle => {
        switch (angle) {
            case 90:
                count90++;
                break;
            case 45:
                count45++;
                break;
            case 0:
                count0++;
                break;
            case -135:
                count135++;
                break;
            default:
                // 他の角度は無視
                break;
        }
    });

// 　平均順位 = (1着回数 x 1 + 2着回数 x 2 + 3着回数 x 3 + 4着回数 x 4 ) / 半荘回数
    let heikin = (count90 * 1 + count45 * 2 + count0 * 3 + count135 * 4) / angles.length;
    console.log(`平均順位: ${heikin}`);
    console.log(
        `一位: ${count90}
二位: ${count45}
三位: ${count0}
四位: ${count135}`
    )
    return { count90, count45, count0, count135 };
}

// jsコンソールに貼って配列を出す