{{- define "operator.name" -}}
distributed-training-operator
{{- end }}

{{- define "operator.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "operator.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "operator.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "operator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "operator.serviceAccountName" -}}
{{ include "operator.fullname" . }}
{{- end }}

{{- define "operator.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end }}
