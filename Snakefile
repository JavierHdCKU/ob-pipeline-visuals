rule plots_flow_metric_plots:
    input:
        metrics = "out/data/data_import/{pre}/{post}/{dataset}.flow_metrics.json.gz"
    output:
        done = "out/data/data_import/{pre}/{post}/plots/flow_metric_plots/default/{dataset}_plots.done"
    params:
        outdir = "out/data/data_import/{pre}/{post}/plots/flow_metric_plots/default"
    conda:
        "plot_env.yml"
    shell:
        r"""
        echo "Input metrics file: {input.metrics}"

        ROOT_DIR=$(dirname {input.metrics})
        echo "Computed ROOT_DIR=$ROOT_DIR"

        mkdir -p {params.outdir}

        python visualize_flow_metrics.py \
            --root $ROOT_DIR \
            --outdir {params.outdir}

        echo "plots generated" > {output.done}
        """
